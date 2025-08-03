use futures::stream::StreamExt;
use std::collections::HashMap;

use ort::{session::Session, value::DynTensor};

pub struct OnnxExecutor {
    pub session: Session,
    pub inputs: flume::Receiver<HashMap<String, DynTensor>>, // <SessionInputs<'a, 'a>>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        while let Some(inputs) = stream.next().await {
            self.session.run(inputs).unwrap();
        }

        println!("Connector disconnected: {}", stream.is_disconnected());
        println!("executor finished");
    }
}
