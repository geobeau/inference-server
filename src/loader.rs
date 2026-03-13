use std::sync::Arc;

use ort::session::{RunOptions, Session};
use smallvec::SmallVec;

use tracing::info;

use crate::{
    metrics::with_local_metrics, scheduler::ModelProxy, tensor::supertensor::SessionValues,
};

pub struct OnnxExecutor {
    pub id: String,
    pub session: Session,
    pub model: Arc<ModelProxy>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        info!(id = %self.id, "executor started");
        let model_name = self.model.model_config.name.clone();

        loop {
            // println!("trying to execute another batch");
            self.model
                .data
                .execute_on_batch(self.id.clone(), async |inputs| {
                    let start = std::time::Instant::now();
                    let run_options: RunOptions = RunOptions::new().unwrap();
                    let session_outputs = self
                        .session
                        .run_async(inputs, &run_options)
                        .unwrap()
                        .await
                        .unwrap();
                    with_local_metrics(|m| {
                        m.observe_model_execution(&model_name, start.elapsed().as_secs_f64());
                    });
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
