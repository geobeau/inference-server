use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering}, usize,
};

use ort::{
    session::Input,
    value::{DynTensor, DynTensorValueType, ValueRef},
};
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
    executor_in_flight: AtomicUsize,
    batch_size: usize,
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
    // Notified executors that work can be done
    executor_notifier: Notify,
    // Notify waiting inference request that a buffer is ready
    infer_full_notifier: Notify,
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
        for i in 0..batch_buffer_capacity {
            buffer.push(TensorBatch::new(i, batch_size, inputs).unwrap());
        }
        BatchRingBuffer {
            buffer,
            tail: AtomicUsize::new(0),
            in_use: AtomicUsize::new(0),
            head: AtomicUsize::new(0),
            batch_size,
            executor_in_flight: AtomicUsize::new(0),
            mask: batch_buffer_capacity - 1,
            executor_notifier: Notify::new(),
            infer_full_notifier: Notify::new(),
        }
    }

    pub fn update_tail_to_next_in_use(&self) {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let in_use = self.in_use.load(Ordering::Acquire);
            if tail == in_use {
                // println!("All buffer put back in queue");
                return;
            }

            let buffer_idx = tail & self.mask;
            // println!(
            //     "tail: updating tail {}",
            //     self.buffer[tail].is_ready_to_use_or_open()
            // );

            if self.buffer[buffer_idx].is_ready_to_use_or_open() {
                match self.tail.compare_exchange_weak(
                    tail,
                    tail+1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        println!("tail: Moved tail to {}", tail+1);
                        self.infer_full_notifier.notify_waiters();
                        continue;
                    }
                    Err(_) => {
                        continue;
                    }
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
            // println!("RingBuffer: appending to {buffer_idx}");
            match buffer.append_on_slot(data, self) {
                Ok(reservation) => {
                    if reservation.should_execute().await {
                        // println!(
                        //     "{} slot {} Attempting to close {buffer_idx}",
                        //     buffer_idx, reservation.slot
                        // );
                        self.try_move_head(head).await;
                        // println!("{} slot {} Closing buffer", buffer_idx, reservation.slot);
                        buffer.close_for_write();
                    }

                    return Ok(buffer.get_data_from_slot(reservation).await);
                }
                Err(slot) => {
                    if slot > self.batch_size * 20 {
                        panic!("Buffer super full, dumping useful data before panic: in flight executor {}\nTail:{}->Use:{}->Head:{}",
                            self.executor_in_flight.load(Ordering::Relaxed),
                            self.tail.load(Ordering::Relaxed),
                            self.in_use.load(Ordering::Relaxed),
                            self.head.load(Ordering::Relaxed),
                        )
                    }
                    // Buffer is full, try to move to next
                    self.try_move_head(head).await;
                }
            }
        }
    }

    async fn try_move_head(&self, current_head: usize) {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head != current_head {
            println!("Head view stale, looping again");
            return;
        }

        // Check if we have space (at least one buffer available)
        if current_head.wrapping_sub(tail) >= self.buffer.len() {
            // self.executor_notifier.notified().await;
            println!("RingBuffer: All buffer full, waiting for buffer capacity: Tail:{tail}->Current Head:{head}");
            self.infer_full_notifier.notified().await;
            return;
        }

        let new_head = head+1;

        // Use CAS to ensure only one thread advances head
        match self.head.compare_exchange_weak(
            current_head,
            new_head,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // println!("RingBuffer: mask {} {}", self.mask, (current_head + 1));
                // println!("RingBuffer: buffer full, moving up {current_head} to {new_head} (tail: {tail})");

                self.executor_notifier.notify_waiters();
                // Successfully moved to next buffer, retry append
            }
            Err(_) => {
                // Another thread moved head, retry with new head
            }
        }
    }

    pub async fn execute_on_batch<F>(&self, f: F)
    where
        F: AsyncFnOnce(HashMap<String, ValueRef<'_, DynTensorValueType>>) -> BatchedOutputs,
    {
        loop {
            // Arm a executor_notifier in case no buffer are available
            let executor_notifier = self.executor_notifier.notified();

            let buffer = match self.get_buffer_to_use() {
                Ok(buffer) => buffer,
                Err(_) => {
                    // println!("+Executor: Parking while waiting for buffer");
                    executor_notifier.await;
                    // println!("+Executor: Notified that buffer can be executed");
                    continue;
                }
            };
            self.executor_in_flight.fetch_add(1, Ordering::Relaxed);
            buffer.execute_on_batch(f).await;
            self.executor_in_flight.fetch_sub(1, Ordering::Relaxed);
            return;
        }
    }

    pub fn get_buffer_to_use(&self) -> Result<&TensorBatch, AppendError> {
        let in_use = self.in_use.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);

        // println!("RingBuffer: in_use:{in_use}/head:{head}/tail:{tail}");
        if in_use == head {
            return Err(AppendError::NoBufferReady);
        }

        // println!("Ringbuffer: giving buffer {in_use}");
        match self.in_use.compare_exchange_weak(
            in_use,
            in_use+1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // Successfully moved to next buffer, retry append
                Ok(&self.buffer[in_use & self.mask])
            }
            Err(_) => {
                // Another thread moved head, retry with new head
                Err(AppendError::NoBufferReady)
            }
        }
    }
}
