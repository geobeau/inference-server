use std::{collections::HashMap, time::Duration};

use futures::StreamExt;
use ort::value::{DynTensorValueType, Value};
use tokio::time::Instant;

use crate::grpc::inference::{model_metadata_response::TensorMetadata, ModelConfig};

pub struct ExecutorEndpoint {
    pub sender: flume::Sender<InferenceRequest>,
}

pub struct Scheduler {
    pub inputs: flume::Receiver<InferenceRequest>,
    pub executors: Vec<ExecutorEndpoint>,
}

pub struct InferenceRequest {
    pub inputs: HashMap<std::string::String, Value<DynTensorValueType>>,
    pub resp_chan: flume::Sender<TracingData>,
    pub tracing: TracingData,
}

#[derive(Debug)]
pub struct TracingData {
    pub start: Instant,
    pub serialization_start: Option<Duration>,
    pub dispatch: Option<Duration>,
    pub scheduling_start: Option<Duration>,
    pub executor_start: Option<Duration>,
    pub send_response: Option<Duration>,
    pub process_response: Option<Duration>,
}

#[derive(Clone)]
pub struct ModelMetadata {
    pub input_meta: Vec<TensorMetadata>,
    pub output_meta: Vec<TensorMetadata>,
}

#[derive(Clone)]
pub struct ModelProxy {
    pub request_sender: flume::Sender<InferenceRequest>,
    pub model_config: ModelConfig,
    pub model_metadata: ModelMetadata,
}

impl Scheduler {
    pub async fn run(&mut self) {
        let mut current_executor = 0;
        while let Some(mut request) = self.inputs.stream().next().await {
            request.tracing.scheduling_start = Some(request.tracing.start.elapsed());
            self.executors[current_executor]
                .sender
                .send_async(request)
                .await
                .unwrap();
            if self.executors[current_executor].sender.is_full() {
                current_executor = (current_executor + 1) % self.executors.len()
            }
        }
        println!("stopping scheduler")
    }
}
