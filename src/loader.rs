use std::sync::Arc;

use ort::session::Session;

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
            self.model
                .data
                .execute_on_batch(|inputs| {
                    let session_outputs = self.session.run(inputs).unwrap();
                    BatchedOutputs::new(session_outputs)
                })
                .await;
        }
    }
}
