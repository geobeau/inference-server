use std::{clone, collections::HashMap};

use futures::stream::StreamExt;

use ort::session::Session;

use crate::scheduler::{InferenceRequest, InferenceResponse};

pub struct OnnxExecutor {
    pub session: Session,
    pub inputs: flume::Receiver<InferenceRequest>, // <SessionInputs<'a, 'a>>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        while let Some(mut req) = stream.next().await {
            req.tracing.executor_start = Some(req.tracing.start.elapsed());
            let mut session_outputs = self.session.run(req.inputs).unwrap();
            req.tracing.send_response = Some(req.tracing.start.elapsed());

            let mut outputs: HashMap<String, ort::value::Value> = HashMap::new();
            let value = session_outputs.remove("output").unwrap();
            outputs.insert(String::from("output"), value);

            let response = InferenceResponse {
                outputs: outputs,
                tracing: req.tracing,
            };
            req.resp_chan.send_async(response).await.unwrap();
        }

        println!("Connector disconnected: {}", stream.is_disconnected());
        println!("executor finished");
    }
}
