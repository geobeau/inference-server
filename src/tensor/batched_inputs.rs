#[cfg(feature = "loom")]
use loom::cell::UnsafeCell;
#[cfg(feature = "loom")]
use loom::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(not(feature = "loom"))]
use std::cell::UnsafeCell;
use std::cmp::min;
use std::time::Duration;
use tokio::time::{sleep_until, Instant};

use ort::session::Input;

use ort::value::{DynTensor, DynTensorValueType, ValueRef};

use tokio::sync::Notify;

use std::collections::HashMap;
use std::sync::atomic::{fence, AtomicUsize, Ordering};

#[cfg(not(feature = "loom"))]
use crate::tensor::batched_tensor::BatchableTensor;
use crate::tensor::batched_tensor::BatchedOutputs;
use crate::tensor::tensor_ringbuffer::BatchRingBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    AllBuffersFull,
    NoBufferReady,
}

pub struct BatchState {
    // 0 -> OpenToWrite
    // 1 -> Written
    // 2 -> Executed
    // 3 -> ReadyToUse
    state: AtomicUsize,
    ready_for_execute_notifier: Notify,
    output_ready_notifier: Notify,
}

impl BatchState {
    fn new() -> BatchState {
        BatchState {
            state: AtomicUsize::new(0),
            ready_for_execute_notifier: Notify::new(),
            output_ready_notifier: Notify::new(),
        }
    }

    pub fn transition_to_written(&self) -> Option<()> {
        match self
            .state
            .compare_exchange_weak(0, 1, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                self.ready_for_execute_notifier.notify_one();
                Some(())
            }
            Err(_) => None,
        }
    }
    pub async fn wait_written(&self) {
        let notified = self.ready_for_execute_notifier.notified();
        if self.state.load(Ordering::SeqCst) != 1 {
            notified.await;
        }
    }
    pub fn transition_to_executed(&self) -> Option<()> {
        match self
            .state
            .compare_exchange_weak(1, 2, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                self.output_ready_notifier.notify_one();
                Some(())
            }
            Err(_) => None,
        }
    }
    pub async fn wait_executed(&self) {
        let notified = self.output_ready_notifier.notified();
        if self.state.load(Ordering::SeqCst) != 2 {
            notified.await;
        }
    }
    pub fn transition_to_ready_to_use(&self) -> Option<()> {
        match self
            .state
            .compare_exchange_weak(2, 0, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Some(()),
            Err(_) => None,
        }
    }
    pub fn is_ready_to_use(&self) -> bool {
        self.state.load(Ordering::SeqCst) == 0
    }
}

// Ensure accounting when tasks are canceled
pub struct WriteReservation<'a> {
    tensor: &'a TensorBatch,
    ring_buffer: &'a BatchRingBuffer,
    pub slot: usize,
}

impl WriteReservation<'_> {
    pub async fn should_execute(&self) -> bool {
        if self.slot + 1 == self.tensor.batch_size {
            self.tensor.full_notifier.notify_one();
            println!("{} slot 0 notified", self.tensor.id);
            return true;
        }
        // If this is the first slot, need to keep track of the timeout
        else if self.slot == 0 {
            println!("{} slot 0 waiting on notifier or timeout", self.tensor.id);
            tokio::select! {
                _ = sleep_until(Instant::now() + Duration::from_millis(10)) => {
                    println!("{} slot 0 buffer timeout", self.tensor.id);
                    return true
                },
                _ = self.tensor.full_notifier.notified() => {
                    println!("{} slot 0 buffer notified", self.tensor.id);
                    return false
                }
            }
        }

        false
    }
}

impl<'a> Drop for WriteReservation<'a> {
    fn drop(&mut self) {
        let slot_written = self.tensor.written_slots.fetch_sub(1, Ordering::Release);
        println!(
            "{} dropping reservation, remaining {}({})",
            self.tensor.id,
            slot_written,
            slot_written - 1
        );
        if slot_written - 1 == 0 {
            self.tensor.reset();
            println!("{} output buffer reset", self.tensor.id);
            self.ring_buffer.update_tail_to_next_in_use();
        }
    }
}

pub struct TensorBatch {
    id: usize,
    #[cfg(feature = "loom")]
    input_data: HashMap<String, Vec<UnsafeCell<Vec<u8>>>>,
    #[cfg(not(feature = "loom"))]
    input_data: HashMap<String, UnsafeCell<BatchableTensor>>,
    output_data: UnsafeCell<Option<BatchedOutputs>>,
    batch_size: usize,
    reserved_slots: AtomicUsize,
    written_slots: AtomicUsize,
    state: BatchState,
    full_notifier: Notify,
}

unsafe impl Send for TensorBatch {}
unsafe impl Sync for TensorBatch {}

impl TensorBatch {
    pub fn close_for_write(&self) {
        loop {
            println!("{} Closing buffer", self.id);
            let reserved_slots = self.reserved_slots.load(Ordering::SeqCst);
            let written_slots = self.written_slots.load(Ordering::SeqCst);
            // Ensure nothing is writing at this moment
            // don't close if something is writing.
            println!(
                "{} {reserved_slots} == min({written_slots}, {})",
                self.id, self.batch_size
            );
            let clamped_slots = min(reserved_slots, self.batch_size);
            if written_slots == clamped_slots {
                // Ensure buffer appear full
                match self.reserved_slots.compare_exchange_weak(
                    reserved_slots,
                    self.batch_size,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        println!("{} Buffer closed", self.id);
                        self.state.transition_to_written().unwrap();
                        return;
                    }
                    Err(_) => {
                        // Something wrote while we were closing
                        println!("{} Collision, retrying", self.id);
                        continue;
                    }
                }
            } else {
                continue;
            }
        }
    }

    fn reserve_write_slot<'a>(
        &'a self,
        slot: usize,
        ring_buffer: &'a BatchRingBuffer,
    ) -> WriteReservation<'a> {
        println!("{} genarating write reservation for {slot}", self.id);
        self.written_slots.fetch_add(1, Ordering::Release);
        WriteReservation {
            tensor: self,
            slot,
            ring_buffer,
        }
    }

    pub fn new(id: usize, batch_size: usize, inputs: &Vec<&Input>) -> Result<TensorBatch, ()> {
        #[cfg(feature = "loom")]
        {
            let mut data = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                data.push(UnsafeCell::new(vec![0; tensor_size]))
            }
            TensorBatch {
                data,
                tensor_size,
                capacity,
                reserved_slots: AtomicUsize::new(0),
                written_slots: AtomicUsize::new(0),
                state: RefCell::new(BatchState::Writtable),
            }
        }
        #[cfg(not(feature = "loom"))]
        {
            let mut input_data = HashMap::new();

            inputs.iter().for_each(|input| {
                let (ty, shape) = match &input.input_type {
                    ort::value::ValueType::Tensor {
                        ty,
                        shape,
                        dimension_symbols: _,
                    } => (ty, shape),
                    ort::value::ValueType::Sequence(_value_type) => todo!(),
                    ort::value::ValueType::Map { key: _, value: _ } => todo!(),
                    ort::value::ValueType::Optional(_value_type) => todo!(),
                };
                let tensor = BatchableTensor::new(*ty, shape, batch_size);
                input_data.insert(input.name.clone(), UnsafeCell::from(tensor));
            });

            Ok(TensorBatch {
                id,
                input_data,
                output_data: UnsafeCell::from(None),
                batch_size,
                reserved_slots: AtomicUsize::new(0),
                written_slots: AtomicUsize::new(0),
                state: BatchState::new(),
                full_notifier: Notify::new(),
            })
        }
    }

    pub fn append_on_slot<'a>(
        &'a self,
        data: &HashMap<String, DynTensor>,
        ring_buffer: &'a BatchRingBuffer,
    ) -> Option<WriteReservation<'a>> {
        let slot = self.append(data)?;

        // Mark this slot as written
        let reserved_slot = self.reserve_write_slot(slot, ring_buffer);
        Some(reserved_slot)
    }

    pub async fn get_data_from_slot(
        &self,
        reservation: WriteReservation<'_>,
    ) -> HashMap<String, DynTensor> {
        self.state.wait_executed().await;
        println!("{} Consuming slot {} ", self.id, reservation.slot);
        let output = self.get_output(reservation.slot);

        println!("{} Dropping reservation {} ", self.id, reservation.slot);
        drop(reservation);
        output
    }

    fn get_output(&self, slot: usize) -> HashMap<String, DynTensor> {
        unsafe {
            let data = (*self.output_data.get()).as_ref().unwrap();
            data.pop_outputs(slot)
        }
    }

    pub fn append(&self, data: &HashMap<String, DynTensor>) -> Option<usize> {
        let slot = self.reserved_slots.fetch_add(1, Ordering::SeqCst);
        if slot >= self.batch_size {
            println!("{} Full {slot}/{}", self.id, self.batch_size);
            if slot > self.batch_size * 20 {
                panic!("Buffer super full, probable deadlock")
            }
            return None;
        }

        data.iter().for_each(|(key, value)| {
            let val = self.input_data.get(key).unwrap();
            #[cfg(feature = "loom")]
            {
                val[slot].with_mut(|ptr| unsafe {
                    let slice = &mut (&mut (*ptr))[0..self.tensor_size];
                    slice.copy_from_slice(data);
                });
            }

            #[cfg(not(feature = "loom"))]
            unsafe {
                // Isolation of portions of the vector is guaranteed by reserved_slots atomics
                (&mut *val.get()).copy_at(slot, value)
            }
        });

        fence(Ordering::Release);

        Some(slot)
    }

    pub async fn execute_on_batch<F>(&self, f: F)
    where
        F: FnOnce(HashMap<String, ValueRef<'_, DynTensorValueType>>) -> BatchedOutputs,
    {
        println!("{} Executor: data ready", self.id);
        let input = self.get_data_view().await;
        unsafe {
            let ptr = self.output_data.get();
            println!("{} Executor: execute", self.id);
            *ptr = Some(f(input));
        }
        fence(Ordering::Release);
        self.state.transition_to_executed().unwrap();
        println!("{} Executor: notifying that output is ready", self.id)
    }

    pub async fn get_data_view(&self) -> HashMap<String, ValueRef<'_, DynTensorValueType>> {
        println!("{} Parking waiting for data ready", self.id);
        self.state.wait_written().await;
        #[cfg(feature = "loom")]
        {
            // For loom, we can't safely return a reference
            // This is a limitation - in real code you'd want proper synchronization
            None
        }

        #[cfg(not(feature = "loom"))]
        {
            unsafe {
                let mut all_inputs = HashMap::new();

                let data = &self.input_data;
                data.iter().for_each(|(name, batch_tensor)| {
                    all_inputs.insert(name.clone(), (*batch_tensor.get()).inner_tensor.view());
                });
                all_inputs
            }
        }
    }

    /// Reset the buffer for reuse
    pub fn reset(&self) {
        self.reserved_slots.store(0, Ordering::Release);
        self.written_slots.store(0, Ordering::Release);
        self.state.transition_to_ready_to_use().unwrap();
    }

    pub fn is_ready_to_use(&self) -> bool {
        self.state.is_ready_to_use()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[cfg(feature = "loom")]
//     use loom::sync::Arc;
//     #[cfg(feature = "loom")]
//     use loom::thread;

//     #[cfg(not(feature = "loom"))]
//     #[test]
//     fn it_works() {
//         let tensor_size = 8;
//         let capacity = 8;
//         let data: Vec<u8> = (0..tensor_size * capacity).map(|i| i as u8).collect();

//         let stacked_tensors = TensorBatch::new(tensor_size, capacity);
//         for i in 0..capacity {
//             let tensor = Tensor::from_array((
//                 [tensor_size],
//                 data[i * tensor_size..(i + 1) * tensor_size].to_vec(),
//             ))
//             .unwrap();
//             stacked_tensors.append(tensor);
//         }
//         let (shape, final_data) = stacked_tensors.get_data().unwrap().extract_tensor();
//         assert_eq!(final_data, &data)
//     }

//     #[test]
//     #[cfg(feature = "loom")]
//     fn loom_concurrent_append_two_threads() {
//         loom::model(|| {
//             let tensor_size = 2;
//             let capacity = 2;

//             let stacked = Arc::new(TensorBatch::new(tensor_size, capacity));

//             let stacked1 = stacked.clone();
//             let stacked2 = stacked.clone();
//             let stacked3 = stacked.clone();
//             // let stacked4 = stacked.clone();

//             let t1 = thread::spawn(move || {
//                 let data = vec![1u8; 2];
//                 stacked1.append(&data)
//             });

//             let t2 = thread::spawn(move || {
//                 let data = vec![2u8; 2];
//                 stacked2.append(&data)
//             });

//             let t3 = thread::spawn(move || {
//                 let data = vec![2u8; 2];
//                 stacked3.append(&data)
//             });

//             // let t4 = thread::spawn(move || {
//             //     let data = vec![2u8; 2];
//             //     stacked4.append(&data)
//             // });

//             let r1 = t1.join().unwrap();
//             let r2 = t2.join().unwrap();
//             let r3 = t3.join().unwrap();
//             // let r4 = t4.join().unwrap();
//             let mut some = 0;
//             let mut none = 0;
//             [r1, r2, r3].into_iter().for_each(|x| match x {
//                 Some(_) => some += 1,
//                 None => none += 1,
//             });
//             assert!(some == 2);
//             assert!(none == 1);

//             // Check that all slots are written
//             assert_eq!(stacked.written_count(), capacity);
//             assert!(stacked.is_ready());
//         });
//     }
// }
