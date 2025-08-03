use std::collections::HashMap;

use futures::StreamExt;
use ort::value::{DynTensorValueType, Value};

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
    pub resp_chan: flume::Sender<()>,
}

pub struct ModelMetadata {
    pub input_meta: Vec<TensorMetadata>,
    pub output_meta: Vec<TensorMetadata>,
}

pub struct ModelProxy {
    pub request_sender: flume::Sender<InferenceRequest>,
    pub model_config: ModelConfig,
    pub model_metadata: ModelMetadata,
}

impl Scheduler {
    pub async fn run(&mut self) {
        let mut current_executor = 0;
        while let Some(inputs) = self.inputs.stream().next().await {
            println!("sending to executor");
            println!("dispatching to {current_executor}");
            self.executors[current_executor]
                .sender
                .send_async(inputs)
                .await
                .unwrap();
            if self.executors[current_executor].sender.is_full() {
                current_executor = (current_executor + 1) % self.executors.len()
            }
        }
        println!("stopping scheduler")
    }
}
