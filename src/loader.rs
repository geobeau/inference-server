use std::sync::Arc;

use ort::session::{RunOptions, Session};
use smallvec::SmallVec;

use crate::{
    scheduler::ModelProxy,
    tensor::supertensor::SessionValues,
};

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
                .execute_on_batch(self.id.clone(), async |inputs| {
                    let run_options: RunOptions = RunOptions::new().unwrap();
                    let session_outputs = self
                        .session
                        .run_async(inputs, &run_options)
                        .unwrap()
                        .await
                        .unwrap();
                    let mut values: smallvec::SmallVec<[ort::value::Value; 4]> =
                        SmallVec::with_capacity(session_outputs.len());
                    session_outputs.into_iter().for_each(|(_, value)| {
                        values.push(value);
                    });
                    SessionValues { values }
                })
                .await;
            // println!("executed batch")
        }
    }
}
