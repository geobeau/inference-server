use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use ort::{
    session::Input,
    value::{DynTensor, DynTensorValueType, ValueRef},
};
use prost::bytes::buf;
use tokio::sync::Notify;

use crate::tensor::{
    batched_inputs::{AppendError, TensorBatch},
    batched_tensor::BatchedOutputs,
};

pub struct BatchRingBuffer {
    buffer: Vec<TensorBatch>,
    tail: AtomicUsize,
    in_use: AtomicUsize,
    head: AtomicUsize,
    // The mask is used for efficient modulo arithmetic to wrap around the ring buffer.
    // The Problem:
    // When you have a ring buffer with n buffers, you need to convert a continuously incrementing index (0, 1, 2, 3, 4, 5...) into a buffer position (0, 1, 2, 3, 0, 1, 2, 3...).
    // Normally you'd use: buffer_idx = head % buffer_count
    // The Optimization:
    // If buffer_count is a power of 2 (e.g., 4, 8, 16), you can replace the slow modulo operation with a fast bitwise AND:
    // If buffer_count = 4 (which is 2^2)
    // mask = 4 - 1 = 3 = 0b0011

    // head = 0  → 0 & 0b0011 = 0
    // head = 1  → 1 & 0b0011 = 1
    // head = 2  → 2 & 0b0011 = 2
    // head = 3  → 3 & 0b0011 = 3
    // head = 4  → 4 & 0b0011 = 0  // Wraps around!
    // head = 5  → 5 & 0b0011 = 1
    // head = 6  → 6 & 0b0011 = 2
    mask: usize,
    notifier: Notify,
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

impl BatchRingBuffer {
    pub fn new(
        batch_buffer_capacity: usize,
        batch_size: usize,
        inputs: &Vec<&Input>,
    ) -> BatchRingBuffer {
        // TODO: check batches_nr is power of 2
        if !is_power_of_two(batch_buffer_capacity) {
            panic!("Buffer is not power of 2: {batch_buffer_capacity}")
        }
        let mut buffer = Vec::with_capacity(batch_buffer_capacity);
        for _ in 0..batch_buffer_capacity {
            buffer.push(TensorBatch::new(batch_size, inputs).unwrap());
        }
        BatchRingBuffer {
            buffer,
            tail: AtomicUsize::new(0),
            in_use: AtomicUsize::new(0),
            head: AtomicUsize::new(0),
            mask: batch_buffer_capacity - 1,
            notifier: Notify::new(),
        }
    }

    fn update_tail_to_next_in_use(&self) {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            if self.buffer[tail].is_ready_to_use() {
                match self.head.compare_exchange_weak(
                    tail,
                    (tail+1) & self.mask,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => continue,
                    Err(_) => continue,
                }
            } else {
                // All ready have been put back in the queue
                return;
            }
        }

    }

    /// Append data to the current buffer, moving to next if full
    pub async fn infer(
        &self,
        data: &HashMap<String, DynTensor>,
    ) -> Result<HashMap<String, DynTensor>, AppendError> {
        loop {
            // head is where the current buffer is, as it is the hottest path.
            // taking an optimistic approach is preferable: try to append the batch
            // to this one and handle the failure with more atomics as needed.
            let head = self.head.load(Ordering::Acquire);

            // Wraps around like modulo, see the definition of self.mask for how it works
            let buffer_idx = head & self.mask;
            let buffer = &self.buffer[buffer_idx];

            // Try to append to current buffer
            match buffer.infer(data).await {
                Some(output) => {
                    if buffer.is_ready_to_use() {
                        self.update_tail_to_next_in_use();
                    }
                    return Ok(output)
                },
                None => {
                    // Buffer is full, try to move to next
                    let tail = self.tail.load(Ordering::Acquire);

                    // Check if we have space (at least one buffer available)
                    let new_head = (head + 1) & self.mask;
                    println!("mask {} {}", self.mask, (head + 1));
                    println!("buffer full, moving up {head} to {new_head})(tail: {tail})");
                    if tail == new_head {
                        return Err(AppendError::AllBuffersFull);
                    }

                    // Use CAS to ensure only one thread advances head
                    match self.head.compare_exchange_weak(
                        head,
                        new_head,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            // Successfully moved to next buffer, retry append
                            continue;
                        }
                        Err(_) => {
                            println!("Collision");
                            // Another thread moved head, retry with new head
                            continue;
                        }
                    }
                }
            }
        }
    }

    pub async fn execute_on_batch<F>(&self, f: F)
    where
        F: FnOnce(HashMap<String, ValueRef<'_, DynTensorValueType>>) -> BatchedOutputs,
    {
        loop {
            // Arm a notifier in case no buffer are available
            let notifier = self.notifier.notified();
            let buffer = match self.get_buffer_to_use() {
                Ok(buffer) => buffer,
                Err(_) => {
                    notifier.await;
                    continue;
                }
            };

            return buffer.execute_on_batch(f);
        }
    }

    pub async fn get_batch_to_execute(
        &self,
    ) -> HashMap<String, ort::value::ValueRef<'_, DynTensorValueType>> {
        loop {
            // Arm a notifier in case no buffer are available
            let notifier = self.notifier.notified();
            let buffer = match self.get_buffer_to_use() {
                Ok(buffer) => buffer,
                Err(_) => {
                    notifier.await;
                    continue;
                }
            };

            return buffer.get_data_view().unwrap();
        }
    }

    pub fn get_buffer_to_use(&self) -> Result<&TensorBatch, AppendError> {
        let in_use = self.in_use.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        if in_use == head {
            return Err(AppendError::NoBufferReady);
        }

        let new_in_use = (in_use + 1) & self.mask;
        match self.in_use.compare_exchange_weak(
            in_use,
            new_in_use,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // Successfully moved to next buffer, retry append

                Ok(&self.buffer[new_in_use])
            }
            Err(_) => {
                println!("Collision");
                // Another thread moved head, retry with new head
                Err(AppendError::NoBufferReady)
            }
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let tensor_size = 1;
//         let capacity = 1;
//         let batches_nr = 4;

//         let ringbatch = BatchRingBuffer::new(batches_nr, tensor_size, capacity);
//         for i in 0..(capacity * batches_nr * 4) {
//             println!("Inserting {i}");
//             ringbatch.append(&vec![0u8; tensor_size]).unwrap();
//             let consume = ringbatch.consume_buffer();
//             println!("Consume {consume:?}")
//         }
//         // let final_data = stacked_tensors.get_data().unwrap();
//         // assert_eq!(final_data, &data)
//     }
// }
