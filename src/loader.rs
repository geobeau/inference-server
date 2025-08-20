use futures::stream::StreamExt;

use http::request;
use ort::{session::{RunOptions, Session}, value::ValueType};
use tracing::Instrument;

use crate::{
    scheduler::{InferenceResponse, Request},
    tensor::{BatchableInputs, BatchedOutputs},
};

pub struct OnnxExecutor {
    pub id: String,
    pub session: Session,
    pub batch_size: usize,
    pub inputs: flume::Receiver<Request>, // <SessionInputs<'a, 'a>>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        let mut batched_requests_chan = Vec::new();
        let mut batched_requests_trace = Vec::new();

        let mut input_batch = BatchableInputs::new(&self.session.inputs, self.batch_size as i64);
        loop {
            // Batch requests
            loop {
                let maybe_request = stream.next().await;
                match maybe_request {
                    Some(request) => {
                        match request {
                    Request::BatchExecute() => {
                        println!("{} incomplete batch: {}", self.id, batched_requests_chan.len());
                        break;
                    }
                    Request::InferenceRequest(mut req) => {
                        // println!("batching: {}", batched_requests_chan.len());
                        req.trace.record_executor_queue();
                        input_batch.append_inputs(req.inputs);
                        batched_requests_chan.push(req.resp_chan);
                        batched_requests_trace.push(req.trace);
                        if batched_requests_chan.len() == self.batch_size {
                            break;
                        }
                    }
                }
                    },
                    None => todo!(),
                }
                
            }
            batched_requests_trace.iter_mut().for_each(|trace| trace.record_executor_start());
            // println!("{} batch: {}", self.id, batched_requests_chan.len());
            let session_outputs = self.session.run(input_batch.session_inputs_view()).unwrap();

            // let mut options = RunOptions::new().unwrap();
            // let run =  self.session.run_async(input_batch.session_inputs_view(), &options).unwrap();
            // println!("{} run: {}", self.id, batched_requests_chan.len());
            // let session_outputs = run.await.unwrap();
            // println!("{} run done: {}", self.id, batched_requests_chan.len());
            let mut batched_tensor = BatchedOutputs::new(session_outputs);

            // println!("{} batch done: {}", self.id, batched_requests_chan.len());

            while let Some(response_chan) = batched_requests_chan.pop() {
                let outputs = batched_tensor.pop_outputs();

                let mut trace = batched_requests_trace.pop().unwrap();
                trace.record_send_response();
                let response = InferenceResponse { outputs, trace };
                response_chan.send_async(response).await.unwrap();
            }

            // println!("{} distpatched all", self.id);
            input_batch.clear();
        }
    }
}
