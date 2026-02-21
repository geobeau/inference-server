use std::time::Duration;

use tokio::time::Instant;

#[derive(Debug)]
pub struct Trace {
    pub start: Instant,
    serialization_start: Option<Duration>,
    dispatch: Option<Duration>,
    scheduling_start: Option<Duration>,
    executor_queue: Option<Duration>,
    executor_start: Option<Duration>,
    send_response: Option<Duration>,
    process_response: Option<Duration>,
}

impl Trace {
    pub fn start() -> Trace {
        let start: Instant = Instant::now();
        Trace {
            start,
            serialization_start: None,
            dispatch: None,
            scheduling_start: None,
            executor_queue: None,
            executor_start: None,
            send_response: None,
            process_response: None,
        }
    }
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn record_serialization_start(&mut self) {
        self.serialization_start = Some(self.start.elapsed())
    }
    pub fn record_dispatch(&mut self) {
        self.dispatch = Some(self.start.elapsed())
    }
    pub fn record_scheduling_start(&mut self) {
        self.scheduling_start = Some(self.start.elapsed())
    }
    pub fn record_executor_start(&mut self) {
        self.executor_start = Some(self.start.elapsed())
    }
    pub fn record_executor_queue(&mut self) {
        self.executor_queue = Some(self.start.elapsed())
    }
    pub fn record_send_response(&mut self) {
        self.send_response = Some(self.start.elapsed())
    }
    pub fn record_process_response(&mut self) {
        self.process_response = Some(self.start.elapsed())
    }
}
