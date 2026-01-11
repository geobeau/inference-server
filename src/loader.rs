use std::sync::Arc;

use ort::session::{RunOptions, Session};

use crate::{scheduler::ModelProxy, tensor::batched_tensor::BatchedOutputs};

pub struct OnnxExecutor {
    pub id: String,
    pub session: Session,
    pub model: Arc<ModelProxy>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");

        loop {
            // println!("trying to execute another batch");
            self.model
                .data
                .execute_on_batch(async |inputs| {
                    // let run_options: RunOptions = RunOptions::new().unwrap();
                    let session_outputs = self.session.run(inputs).unwrap();
                    BatchedOutputs::new(session_outputs)
                })
                .await;
            // println!("executed batch")
        }
    }
}
