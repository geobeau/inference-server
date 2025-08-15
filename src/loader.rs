use futures::stream::StreamExt;

use ort::{session::Session, value::ValueType};

use crate::{
    scheduler::{InferenceRequest, InferenceResponse},
    tensor::{BatchableInputs, BatchedOutputs},
};

pub struct OnnxExecutor {
    pub session: Session,
    pub inputs: flume::Receiver<InferenceRequest>, // <SessionInputs<'a, 'a>>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        let batch_size = 2;
        let mut batched_requests_chan = Vec::new();
        let mut batched_requests_trace = Vec::new();

        let mut input_batch = BatchableInputs::new(&self.session.inputs, batch_size as i64);

        let (_ty, mut shape) = match &self.session.outputs[0].output_type {
            ValueType::Tensor { ty, shape, .. } => (*ty, shape.clone()),
            ValueType::Sequence(_) => todo!(),
            ValueType::Map { .. } => todo!(),
            ValueType::Optional(_) => todo!(),
        };
        shape[0] = batch_size as i64;
        println!("{:?}", shape);
        loop {
            // Batch requests
            while let Some(mut req) = stream.next().await {
                req.trace.record_executor_start();
                input_batch.append_inputs(req.inputs);
                batched_requests_chan.push(req.resp_chan);
                batched_requests_trace.push(req.trace);
                if batched_requests_chan.len() == batch_size {
                    break;
                }
            }

            let session_outputs = self.session.run(input_batch.session_inputs_view()).unwrap();
            let mut batched_tensor = BatchedOutputs::new(session_outputs);

            while let Some(response_chan) = batched_requests_chan.pop() {
                let outputs = batched_tensor.pop_outputs();

                let mut trace = batched_requests_trace.pop().unwrap();
                trace.record_send_response();
                let response = InferenceResponse { outputs, trace };
                response_chan.send_async(response).await.unwrap();
            }
            input_batch.clear();
        }
    }
}
