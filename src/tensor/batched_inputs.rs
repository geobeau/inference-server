#[cfg(feature = "loom")]
use loom::cell::UnsafeCell;
#[cfg(feature = "loom")]
use loom::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(not(feature = "loom"))]
use std::cell::UnsafeCell;

use ort::session::Input;

use ort::value::{DynTensor, DynTensorValueType, ValueRef};

use tokio::sync::Notify;

use std::cell::RefCell;

use std::collections::HashMap;
use std::sync::atomic::{fence, AtomicUsize, Ordering};

#[cfg(not(feature = "loom"))]
use crate::tensor::batched_tensor::BatchableTensor;
use crate::tensor::batched_tensor::BatchedOutputs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    AllBuffersFull,
    NoBufferReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchState {
    Writtable,
    Written,
    Executed,
    ReadyToUse,
}

pub struct TensorBatch {
    #[cfg(feature = "loom")]
    input_data: HashMap<String, Vec<UnsafeCell<Vec<u8>>>>,
    #[cfg(not(feature = "loom"))]
    input_data: HashMap<String, UnsafeCell<BatchableTensor>>,
    output_data: UnsafeCell<Option<BatchedOutputs>>,
    batch_size: usize,
    reserved_slots: AtomicUsize,
    written_slots: AtomicUsize,
    state: RefCell<BatchState>,
    notifier: Notify,
}

unsafe impl Send for TensorBatch {}
unsafe impl Sync for TensorBatch {}

impl TensorBatch {
    pub fn new(batch_size: usize, inputs: &Vec<&Input>) -> Result<TensorBatch, ()> {
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
                let tensor = BatchableTensor::new(*ty, shape);
                input_data.insert(input.name.clone(), UnsafeCell::from(tensor));
            });

            Ok(TensorBatch {
                input_data,
                output_data: UnsafeCell::from(None),
                batch_size,
                reserved_slots: AtomicUsize::new(0),
                written_slots: AtomicUsize::new(0),
                state: RefCell::new(BatchState::Writtable),
                notifier: Notify::new(),
            })
        }
    }

    pub async fn infer(
        &self,
        data: &HashMap<String, DynTensor>,
    ) -> Option<HashMap<String, DynTensor>> {
        let slot = match self.append(data) {
            Some(slot) => slot,
            None => return None,
        };

        let notifier = self.notifier.notified();

        // Mark this slot as written
        let slot_written = self.written_slots.fetch_add(1, Ordering::Release);
        if slot_written + 1 == self.batch_size {
            self.state.replace(BatchState::Written);
        }

        notifier.await;
        let output = self.get_output(slot);

        let slot_written = self.written_slots.fetch_sub(1, Ordering::Release);
        if slot_written - 1 == self.batch_size {
            self.reset();
            self.state.replace(BatchState::ReadyToUse);
        }
        return Some(output)
    }

    pub fn get_output(&self, slot: usize) -> HashMap<String, DynTensor> {
        unsafe {
            let data = (*self.output_data.get()).as_ref().unwrap();
            return data.pop_outputs(slot);
        }
    }

    pub fn append(&self, data: &HashMap<String, DynTensor>) -> Option<usize> {
        let slot = self.reserved_slots.fetch_add(1, Ordering::SeqCst);
        if slot >= self.batch_size {
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

    pub fn execute_on_batch<F>(&self, f: F)
    where
        F: FnOnce(HashMap<String, ValueRef<'_, DynTensorValueType>>) -> BatchedOutputs,
    {
        let input = self.get_data_view().unwrap();
        unsafe {
            let ptr = self.output_data.get();
            *ptr = Some(f(input));
        }
        self.state.replace(BatchState::Executed);
        fence(Ordering::Release);
        self.notifier.notify_waiters();
    }

    pub fn get_data_view(&self) -> Option<HashMap<String, ValueRef<'_, DynTensorValueType>>> {
        if self.written_slots.load(Ordering::Acquire) == self.batch_size {
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
                    Some(all_inputs)
                }
            }
        } else {
            None
        }
    }

    /// Reset the buffer for reuse
    pub fn reset(&self) {
        self.reserved_slots.store(0, Ordering::Release);
        self.written_slots.store(0, Ordering::Release);
    }

    pub fn is_ready_to_use(&self) -> bool {
        *self.state.borrow() == BatchState::ReadyToUse
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
