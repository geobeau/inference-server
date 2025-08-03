use futures::stream::StreamExt;

use ort::session::Session;

use crate::scheduler::InferenceRequest;

pub struct OnnxExecutor {
    pub session: Session,
    pub inputs: flume::Receiver<InferenceRequest>, // <SessionInputs<'a, 'a>>,
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        while let Some(req) = stream.next().await {
            self.session.run(req.inputs).unwrap();
            req.resp_chan.send_async(()).await.unwrap();
        }

        println!("Connector disconnected: {}", stream.is_disconnected());
        println!("executor finished");
    }
}
