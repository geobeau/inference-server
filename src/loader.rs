use std::{clone, collections::HashMap};

use futures::stream::StreamExt;

use ndarray::{stack, Axis};
use ort::{memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType}, session::Session, value::{DynTensorValueType, Tensor, TensorArrayData, Value}};

use crate::scheduler::{InferenceRequest, InferenceResponse};

pub struct OnnxExecutor {
    pub session: Session,
    pub inputs: flume::Receiver<InferenceRequest>, // <SessionInputs<'a, 'a>>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        let batch_size = 64;
        let mut inputs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<ndarray::IxDynImpl>>> = Vec::new();
        let mut batched_requests_chan = Vec::new();
        let mut batched_requests_tracing = Vec::new();
        loop {
            // Batch requests
            while let Some(mut req) = stream.next().await {
                req.tracing.executor_start = Some(req.tracing.start.elapsed());
                req.inputs.iter().for_each(|(key, value)| {
                    let array: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<ndarray::IxDynImpl>> = value.try_extract_array::<i64>().unwrap().to_owned();
                    inputs.push(array);
                });

                batched_requests_chan.push(req.resp_chan);
                batched_requests_tracing.push(req.tracing);
                if batched_requests_chan.len() == batch_size {
                    break
                }
            }

            // Stack inputs
            let as_view: Vec<_> = inputs.iter().map(|input| input.view().remove_axis(Axis(0))).collect();
            let stack = stack(Axis(0), &as_view).unwrap();
            let stacked_input = Tensor::from_array(stack).unwrap();
            let mut all_inputs = HashMap::new();
            all_inputs.insert("input", stacked_input);
            let mut session_outputs = self.session.run(all_inputs).unwrap();


            let output = session_outputs.remove("output").unwrap();
            let array: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<ndarray::IxDynImpl>> = output.try_extract_array::<f64>().unwrap();
            let mut all_outputs = Vec::new(); 
            array.axis_iter(Axis(0)).for_each(|val| {
                let mut outputs: HashMap<String, Value<DynTensorValueType>> = HashMap::new();

                outputs.insert(String::from("output"), Tensor::from_array(val.to_owned()).unwrap().upcast());
                all_outputs.push(outputs);
                
            });
            while let Some(outputs) = all_outputs.pop() {
                let trace = batched_requests_tracing.pop().unwrap();
                let response = InferenceResponse {
                    outputs: outputs,
                    tracing: trace,
                };
                batched_requests_chan.pop().unwrap().send_async(response).await.unwrap();
            }
            inputs.clear();
        }
    }
}
