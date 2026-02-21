use std::{
    cell::{Ref, RefCell, UnsafeCell},
    collections::HashMap,
    marker,
    sync::{
        atomic::{fence, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
    usize,
};

use arc_swap::ArcSwap;
use futures::FutureExt;
use compio::runtime::time::sleep_until;
use std::time::Instant;
use ort::{
    memory::Allocator, value::{DynTensor, DynTensorValueType, Outlet, ValueRef}
};
use tokio::{
    sync::{futures::Notified, Notify, RwLock},
};

use crate::tensor::batched_tensor::{BatchableTensor, BatchedOutputs};

const HALF_RANGE: usize = usize::MAX / 2;

pub struct Trace {
    pub batch_first_open: RefCell<std::time::Instant>,
    pub batch_complete: RefCell<std::time::Duration>,
    batch_inference_start: RefCell<std::time::Duration>,
    batch_inference_done: RefCell<std::time::Duration>,
    batch_released: RefCell<std::time::Duration>,
}

impl Trace {
    fn print_debug(&self) {
        println!("---\ntime to complete batch {:?}\n picked by executor {:?}\n inference duration {:?}\n time to gather output {:?}", 
            self.batch_complete.borrow(),
            *self.batch_inference_start.borrow() - *self.batch_complete.borrow(),
            *self.batch_inference_done.borrow() - *self.batch_inference_start.borrow(),
            *self.batch_released.borrow() - *self.batch_inference_done.borrow(),
        )
    }
}

struct DataTracker {
    // Dirty buffer should not be reused
    dirty: AtomicUsize,
    written_slots: AtomicUsize,
    deadline: ArcSwap<Option<Instant>>,
    executor_notifier: ArcSwap<Notify>,
    // TODO: make a proper error state
    output: ArcSwap<ArcSwap<Result<BatchedOutputs, usize>>>,
    response_ready_notifier: Notify,
}

// Ensure accounting when tasks are canceled
pub struct WriteReservation<'a> {
    tracker: &'a DataTracker,
    output: Arc<ArcSwap<Result<BatchedOutputs, usize>>>,
    response_ready_notified: Notified<'a>,
    slot: usize,
}

static IN_FLIGHT_REQ: AtomicUsize = AtomicUsize::new(0);

impl WriteReservation<'_> {
    fn new(tracker: &DataTracker, slot: usize) -> WriteReservation<'_> {
        let response_ready_notified = tracker.response_ready_notifier.notified();
        let output_arc = tracker.output.load_full().clone();
        tracker.written_slots.fetch_add(1, Ordering::Release);
        WriteReservation {
            tracker: tracker,
            slot,
            output: output_arc,
            response_ready_notified,
        }
    }

    async fn get_result(self) -> HashMap<String, DynTensor> {
        // IN_FLIGHT_REQ.fetch_add(1, Ordering::Relaxed);
        self.response_ready_notified.await;
        // let in_flight = IN_FLIGHT_REQ.fetch_sub(1, Ordering::Relaxed);
        // println!("in flight: {}", in_flight - 1);
        let output_ref = self.output.load();
        let mut slot_output = HashMap::new();
        match &output_ref.as_ref() {
            Ok(output) => {
                output.outputs.iter().for_each(|(name, batch_tensor)| {
                    slot_output.insert(name.clone(), batch_tensor.pop_at(self.slot));
                });
            }
            Err(_) => todo!(),
        }
        return slot_output;
    }
}

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Useful abstraction to get the correct position within the buffer
/// Maybe it should be a macro?
pub struct RingBufferIndex<'a> {
    /// Absolute index on the ringbuffer
    index: usize,
    atomic_ref: &'a PaddedAtomic,
}

impl<'a> RingBufferIndex<'a> {
    fn new(index: usize, atomic_ref: &PaddedAtomic) -> RingBufferIndex {
        return RingBufferIndex { index, atomic_ref };
    }
    /// Get the absolute index, return the raw index, useful for operating on the ring
    /// itself
    fn as_absolute_index(&self) -> usize {
        return self.index;
    }

    /// Return the batch_slot 0 of the next batch in absolute index
    fn as_absolute_batch_higher_bound(&self) -> usize {
        self.as_absolute_batch_lower_bound()
            .wrapping_add(self.atomic_ref.batch_size)
    }
    /// Return the batch_slot 0 of the current batch in absolute index
    fn as_absolute_batch_lower_bound(&self) -> usize {
        self.index.wrapping_sub(self.as_batch_slot_id())
    }

    /// Get the batch id: the idx of the batch that is concerned by this index
    fn as_batch_id(&self) -> usize {
        // println!("{}: {} / {} -> {}", self.index , self.index & self.ring_buffer.ring_mask, self.ring_buffer.batch_size, (self.index & self.ring_buffer.ring_mask) / self.ring_buffer.batch_size);
        (self.index & self.atomic_ref.ring_mask) / self.atomic_ref.batch_size
    }

    /// Get the batch slot id: the idx of the slot relative to the batch id
    fn as_batch_slot_id(&self) -> usize {
        self.index & self.atomic_ref.batch_mask
    }

    pub fn wrapping_sub(&self, other: &RingBufferIndex) -> usize {
        self.index.wrapping_sub(other.index)
    }

    pub fn wrapping_add(&self, rhs: usize) -> usize {
        self.index.wrapping_add(rhs)
    }
}

unsafe impl Send for SuperTensorBuffer {}
unsafe impl Sync for SuperTensorBuffer {}

/// To avoid false sharing, padding the atomic with the worst case cache line
/// This also helps operating safely around the heads/tail
#[repr(align(128))]
struct PaddedAtomic {
    value: AtomicUsize,
    // Mask for the full ring: batch_size * capacity
    ring_mask: usize,
    // Mask to get position within the batch
    batch_mask: usize,
    batch_size: usize,
}

impl PaddedAtomic {
    pub fn load(&self, order: Ordering) -> RingBufferIndex {
        RingBufferIndex::new(self.value.load(order), self)
    }

    pub fn compare_exchange_weak(&self, current: usize, new: usize) -> Result<usize, usize> {
        self.value
            .compare_exchange_weak(current, new, Ordering::SeqCst, Ordering::Relaxed)
    }
}

pub struct SuperTensorBuffer {
    input_tensors: HashMap<String, Vec<UnsafeCell<BatchableTensor>>>,
    trackers: Vec<DataTracker>,
    batch_size: usize,
    capacity: usize,
    head: PaddedAtomic,
    executor_head: PaddedAtomic,
    tail: PaddedAtomic,
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
    // Mask for the full ring: batch_size * capacity
    ring_mask: usize,
    // Mask to get position within the batch
    batch_mask: usize,
    executor_full_notifier: Notify,
    infer_full_notifier: Notify,
}

impl SuperTensorBuffer {
    pub fn new(
        capacity: usize,
        batch_size: usize,
        inputs: &Vec<&Outlet>,
        allocator: &Allocator,
    ) -> Result<SuperTensorBuffer, ()> {
        {
            if !is_power_of_two(capacity) {
                panic!("Buffer is not power of 2: {capacity}")
            }
            let mut input_tensors = HashMap::new();

            inputs.iter().for_each(|input| {
                let (ty, shape) = match &input.dtype() {
                    ort::value::ValueType::Tensor {
                        ty,
                        shape,
                        dimension_symbols: _,
                    } => (ty, shape),
                    ort::value::ValueType::Sequence(_value_type) => todo!(),
                    ort::value::ValueType::Map { key: _, value: _ } => todo!(),
                    ort::value::ValueType::Optional(_value_type) => todo!(),
                };
                let mut batched_tensors = Vec::with_capacity(capacity);
                for i in 0..capacity {
                    batched_tensors.push(UnsafeCell::from(BatchableTensor::new(
                        ty.clone(), shape, batch_size, &Allocator::default(),
                    )));
                }

                input_tensors.insert(input.name().to_string(), batched_tensors);
            });

            let mut trackers = Vec::with_capacity(capacity);
            for i in 0..capacity {
                trackers.push(DataTracker {
                    dirty: AtomicUsize::new(0),
                    written_slots: AtomicUsize::new(0),
                    deadline: ArcSwap::from_pointee(None),
                    output: ArcSwap::from(Arc::from(ArcSwap::from(Arc::from(Err(0))))),
                    response_ready_notifier: Notify::new(),
                    executor_notifier: ArcSwap::from(Arc::from(Notify::new())),
                });
            }
            let ring_mask = (capacity * batch_size) - 1;
            let batch_mask = batch_size - 1;

            Ok(SuperTensorBuffer {
                input_tensors,
                trackers,
                batch_size,
                head: PaddedAtomic {
                    value: AtomicUsize::new(0),
                    ring_mask,
                    batch_mask,
                    batch_size,
                },
                executor_head: PaddedAtomic {
                    value: AtomicUsize::new(0),
                    ring_mask,
                    batch_mask,
                    batch_size,
                },
                tail: PaddedAtomic {
                    value: AtomicUsize::new(0),
                    ring_mask,
                    batch_mask,
                    batch_size,
                },
                capacity,
                ring_mask,
                batch_mask,
                executor_full_notifier: Notify::new(),
                infer_full_notifier: Notify::new(),
            })
        }
    }

    pub async fn infer(
        &self,
        data: &HashMap<String, DynTensor>,
    ) -> Result<HashMap<String, DynTensor>, usize> {
        loop {
            let current_head = self.head.load(Ordering::Relaxed);
            let current_tail = self.tail.load(Ordering::Acquire);

            // Check if the ring is full
            if current_head.wrapping_sub(&current_tail) >= self.capacity * self.batch_size {
                // println!("Buffer full, yielding");
                self.infer_full_notifier.notified().await;
                continue;
            }

            match self.head.compare_exchange_weak(
                current_head.as_absolute_index(),
                current_head.wrapping_add(1),
            ) {
                Ok(_) => {
                    let write_reservation = self.insert_tensors_at(current_head, data);
                    return Ok(write_reservation.get_result().await);
                }
                Err(_) => {
                    // Another producer won the race, retry the check
                    continue;
                }
            }
        }
    }

    fn insert_tensors_at(
        &self,
        idx: RingBufferIndex,
        data: &HashMap<String, DynTensor>,
    ) -> WriteReservation {
        // batch_slot is the index of the slot within a batch
        // Used to track if it's the first or last reservation on the batch
        // println!(
        //     "Inserting in {} at -> {}",
        //     idx.as_batch_id(),
        //     idx.as_batch_slot_id()
        // );
        let batch_slot = idx.as_batch_slot_id();
        data.iter().for_each(|(key, value)| {
            let tensors = self.input_tensors.get(key).unwrap();
            let val = tensors.get(idx.as_batch_id()).unwrap();
            unsafe {
                // Isolation of portions of the vector is guaranteed by reserved_slots atomics
                (&mut *val.get()).copy_at(batch_slot, value)
            }
        });
        if batch_slot == 0 {
            let tracker = self.trackers.get(idx.as_batch_id()).unwrap();
            let deadline = Instant::now() + Duration::from_millis(2);
            tracker.dirty.store(1, Ordering::Relaxed);
            // println!("{}> writing deadline", idx.as_batch_id());
            tracker.deadline.store(Arc::from(Some(deadline)));
        }
        fence(Ordering::Release);

        let tracker = self.trackers.get(idx.as_batch_id()).unwrap();
        let reservation = WriteReservation::new(tracker, batch_slot);
        // If the batch is going to be completed, awaken the executor
        if batch_slot == (self.batch_size - 1) || batch_slot == 0 {
            // println!("Notifying executor on {}", idx.as_batch_id());
            tracker.executor_notifier.load_full().notify_one();
        }
        return reservation;
    }

    pub async fn execute_on_batch<F>(&self, id: String, f: F)
    where
        F: AsyncFnOnce(HashMap<String, ValueRef<'_, DynTensorValueType>>) -> BatchedOutputs,
    {
        let mut current_executor_idx; // Defined as RingBufferIndex
        loop {
            current_executor_idx = self.executor_head.load(Ordering::Acquire);
            let current_tail = self.tail.load(Ordering::Acquire);

            // Check if the ring is full: (executor_head - tail) >= total_capacity
            if current_executor_idx.wrapping_sub(&current_tail) >= self.capacity * self.batch_size {
                self.executor_full_notifier.notified().await;
                continue;
            }

            match self.executor_head.compare_exchange_weak(
                current_executor_idx.as_absolute_index(),
                current_executor_idx
                    .as_absolute_index()
                    .wrapping_add(self.batch_size),
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        let tracker = self
            .trackers
            .get(current_executor_idx.as_batch_id())
            .unwrap();
        tracker.executor_notifier.load().notified().await;

        let notifier = tracker.executor_notifier.load();
        let notified_full = notifier.notified();
        let head = self.head.load(Ordering::Acquire);

        // Wrapping safe: head < current_executor_idx.as_absolute_batch_higher_bound()
        // Logic: Distance from head to higher_bound is > 0 and < HALF_RANGE
        let dist_to_higher = current_executor_idx
            .as_absolute_batch_higher_bound()
            .wrapping_sub(head.as_absolute_index());
        if dist_to_higher > 0 && dist_to_higher < HALF_RANGE {
            let maybe_deadline = tracker.deadline.load_full();
            if let Some(deadline) = *maybe_deadline {
                tokio::select! {
                    _ = sleep_until(deadline) => {},
                    _ = notified_full => {},
                }
            }
        }

        self.seal_current_batch(tracker, &current_executor_idx);
        self.execute_current_batch(f, tracker, &current_executor_idx)
            .await;
        self.reset_batch(tracker);
        self.move_tail_to_next_non_dirty_buffer();
    }

    async fn execute_current_batch<F>(
        &self,
        f: F,
        tracker: &DataTracker,
        current_executor_idx: &RingBufferIndex<'_>,
    ) where
        F: AsyncFnOnce(HashMap<String, ValueRef<'_, DynTensorValueType>>) -> BatchedOutputs,
    {
        let input = self.get_data_view(current_executor_idx);
        let result = f(input).await;
        // Put the results in the arc output, to be dispatched to consumers
        tracker.output.load_full().store(Arc::from(Ok(result)));
        tracker.response_ready_notifier.notify_waiters();
    }

    fn reset_batch(&self, tracker: &DataTracker) {
        tracker.deadline.store(Arc::from(None));
        tracker
            .output
            .store(Arc::from(ArcSwap::from(Arc::from(Err(0)))));
        tracker.written_slots.store(0, Ordering::Relaxed);
        tracker.dirty.store(2, Ordering::Relaxed);
        // executor_notifier can be called twice, recreating one avoid keeping older permits
        tracker.executor_notifier.store(Arc::from(Notify::new()));
    }
    fn move_tail_to_next_non_dirty_buffer(&self) {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let head = self.head.load(Ordering::Acquire);

            // Wrapping safe: tail.higher_bound >= head
            if tail
                .as_absolute_batch_higher_bound()
                .wrapping_sub(head.as_absolute_index())
                < HALF_RANGE
            {
                break;
            }

            let dirty_state = &self.trackers[tail.as_batch_id()].dirty;
            let dirty = dirty_state.load(Ordering::Acquire);
            if dirty == 2 {
                if dirty_state
                    .compare_exchange_weak(dirty, 0, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    if self
                        .tail
                        .compare_exchange_weak(
                            tail.as_absolute_index(),
                            tail.as_absolute_batch_higher_bound(),
                        )
                        .is_ok()
                    {
                        self.executor_full_notifier.notify_waiters();
                        self.infer_full_notifier.notify_waiters();
                        continue;
                    }
                }
            }
            break;
        }
    }
    pub fn get_data_view(
        &self,
        current_executor_idx: &RingBufferIndex,
    ) -> HashMap<String, ValueRef<'_, DynTensorValueType>> {
        unsafe {
            let mut all_inputs = HashMap::new();
            let batch_id = current_executor_idx.as_batch_id();

            self.input_tensors.iter().for_each(|(name, batch_tensors)| {
                let batch_tensor = batch_tensors.get(batch_id).unwrap();
                all_inputs.insert(name.clone(), (*batch_tensor.get()).inner_tensor.view());
            });
            all_inputs
        }
    }

    fn seal_current_batch(&self, tracker: &DataTracker, current_executor_idx: &RingBufferIndex) {
        let mut remaining_open_slots;
        loop {
            remaining_open_slots = 0;
            let head = self.head.load(Ordering::Acquire);

            // head must be ahead of or equal to the lower bound of the batch we are sealing
            assert!(
                head.as_absolute_index()
                    .wrapping_sub(current_executor_idx.as_absolute_batch_lower_bound())
                    < HALF_RANGE,
                "Batch is being sealed but not even written yet"
            );

            // Check if head is still within this batch: head < higher_bound
            let dist_to_higher = current_executor_idx
                .as_absolute_batch_higher_bound()
                .wrapping_sub(head.as_absolute_index());
            if dist_to_higher > 0 && dist_to_higher < HALF_RANGE {
                remaining_open_slots = dist_to_higher;
                if self
                    .head
                    .compare_exchange_weak(
                        head.as_absolute_index(),
                        head.as_absolute_index().wrapping_add(remaining_open_slots),
                    )
                    .is_ok()
                {
                    break;
                }
                continue;
            }
            break;
        }

        loop {
            let written_slots = tracker.written_slots.load(Ordering::Acquire);
            // println!("{}> remaining_open_slots open_slots:{remaining_open_slots}, written:{written_slots}; expected_written: {} - {}", current_executor_idx.as_batch_id(), self.batch_size , remaining_open_slots);
            let expected_written_slots = self.batch_size - remaining_open_slots;
            if written_slots == expected_written_slots {
                return;
            } else if written_slots < expected_written_slots {
                // Some writes are in progress, spinning until they are visible
                continue;
            } else {
                panic!(
                    "{}> written_slots ({}) is higher than expected_written_slots({})",
                    current_executor_idx.as_batch_id(),
                    written_slots,
                    expected_written_slots
                )
            }
        }
    }
}
