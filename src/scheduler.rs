use std::collections::HashMap;

use futures::StreamExt;
use ort::value::{DynTensorValueType, Value};

pub struct ExecutorEndpoint {
    pub sender: flume::Sender<HashMap<std::string::String, Value<DynTensorValueType>>>,
}

pub struct Scheduler {
    pub inputs: flume::Receiver<HashMap<std::string::String, Value<DynTensorValueType>>>,
    pub executors: Vec<ExecutorEndpoint>,
}

impl Scheduler {
    pub async fn run(&mut self) {
        let mut current_executor = 0;
        while let Some(inputs) = self.inputs.stream().next().await {
            println!("sending to executor");
            self.executors[current_executor]
                .sender
                .send(inputs)
                .unwrap();
            if self.executors[current_executor].sender.is_full() {
                current_executor += 1 % self.executors.len()
            }
        }
        println!("stopping scheduler")
    }
}
