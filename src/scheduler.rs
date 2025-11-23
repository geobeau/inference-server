// use std::{
//     cmp::max,
//     collections::HashMap,
//     ops::Sub,
//     time::{Duration, Instant},
// };

// use futures::StreamExt;
// use ort::value::{DynTensorValueType, Value};
// use tokio::time::timeout;

// use crate::{
//     grpc::inference::{model_metadata_response::TensorMetadata, ModelConfig},
//     tensor::tensor_ringbuffer::BatchRingBuffer,
//     tracing::Trace,
// };

// pub struct ExecutorEndpoint {
//     pub sender: flume::Sender<Request>,
// }

// pub struct Scheduler {
//     pub inputs: flume::Receiver<InferenceRequest>,
//     pub executors: Vec<ExecutorEndpoint>,
//     pub max_queue_time: Duration,
//     pub batch_size: i32,
// }

// pub enum Request {
//     // Trigger the batch execution now
//     BatchExecute(),
//     InferenceRequest(InferenceRequest),
// }

// pub struct InferenceRequest {
//     pub inputs: HashMap<std::string::String, Value<DynTensorValueType>>,
//     pub resp_chan: flume::Sender<InferenceResponse>,
//     pub trace: Trace,
// }

// pub struct InferenceResponse {
//     pub outputs: HashMap<std::string::String, Value<DynTensorValueType>>,
//     pub trace: Trace,
// }

use std::collections::{HashMap, HashSet};

use ort::tensor::Shape;

use crate::{
    grpc::inference::{model_metadata_response::TensorMetadata, ModelConfig},
    tensor::tensor_ringbuffer::BatchRingBuffer,
};

#[derive(Clone)]
pub struct ModelMetadata {
    pub input_meta: Vec<TensorMetadata>,
    pub output_meta: Vec<TensorMetadata>,
    pub input_set: HashMap<String, Shape>,
}

pub struct ModelProxy {
    pub data: BatchRingBuffer,
    pub model_config: ModelConfig,
    pub model_metadata: ModelMetadata,
}

// impl Scheduler {
//     pub async fn run(&mut self) {
//         let mut current_executor = 0;
//         let mut current_batched_records = 0;
//         let mut oldest_record = Instant::now();
//         let long_sleep = Duration::from_secs(1);
//         let mut queue_timeout;
//         loop {
//             if current_batched_records == 0 {
//                 // If no records pending, it's ok to pause the scheduler for a while
//                 queue_timeout = long_sleep
//             } else {
//                 queue_timeout = Duration::from_micros(max(
//                     self.max_queue_time.as_micros() as i128
//                         - oldest_record.elapsed().as_micros() as i128,
//                     0i128,
//                 ) as u64);
//             }

//             // let maybe_request = timeout(queue_timeout, self.inputs.stream().next()).await;
//             let maybe_request: Result<Option<InferenceRequest>, tokio::time::error::Elapsed> =
//                 Ok(self.inputs.stream().next().await);
//             match maybe_request {
//                 Ok(req) => {
//                     match req {
//                         Some(mut req) => {
//                             if current_batched_records == 0 {
//                                 // Start keeping track for the first record of the batch
//                                 oldest_record = Instant::now();
//                             }
//                             // println!("scheduling {} {}", current_executor, self.executors[current_executor].sender.len());
//                             req.trace.record_scheduling_start();
//                             self.executors[current_executor]
//                                 .sender
//                                 .send_async(Request::InferenceRequest(req))
//                                 .await
//                                 .unwrap();
//                             // println!("scheduled {}", self.executors[current_executor].sender.len());
//                             current_batched_records += 1;
//                             if current_batched_records >= self.batch_size {
//                                 current_executor = (current_executor + 1) % self.executors.len();
//                                 current_batched_records = 0;
//                                 oldest_record = Instant::now();
//                             }
//                         }
//                         None => {
//                             // Is the chan closed?
//                             todo!();
//                             break;
//                         }
//                     }
//                 }
//                 // Error means the oldest req is above the max_queue_time, batch should be submitted
//                 Err(_) => {
//                     println!(
//                         "timeout: {:?} - {:?}: {:?}",
//                         self.max_queue_time,
//                         oldest_record.elapsed(),
//                         queue_timeout
//                     );
//                     if current_batched_records > 0 {
//                         self.executors[current_executor]
//                             .sender
//                             .send_async(Request::BatchExecute())
//                             .await
//                             .unwrap();
//                         current_executor = (current_executor + 1) % self.executors.len();
//                         current_batched_records = 0;
//                     }

//                     oldest_record = Instant::now();
//                 }
//             }
//         }
//         println!("stopping scheduler")
//     }
// }
