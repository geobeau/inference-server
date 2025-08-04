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
        while let Some(mut req) = stream.next().await {
            req.tracing.executor_start = Some(req.tracing.start.elapsed());
            self.session.run(req.inputs).unwrap();
            req.tracing.send_response = Some(req.tracing.start.elapsed());
            req.resp_chan.send_async(req.tracing).await.unwrap();
        }

        println!("Connector disconnected: {}", stream.is_disconnected());
        println!("executor finished");
    }
}
