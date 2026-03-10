use std::cell::RefCell;
use std::time::Duration;

use tokio::time::Instant;

use tracing::debug;

use crate::metrics::LocalMetrics;

pub struct ClientTrace {
    pub start: Instant,
    model_proxy_aquired: Option<Duration>,
    serialization_done: Option<Duration>,
    inference_in_queue: Option<Duration>,
    output_processed: Option<Duration>,
}

impl ClientTrace {
    pub fn start() -> ClientTrace {
        ClientTrace {
            start: Instant::now(),
            model_proxy_aquired: None,
            serialization_done: None,
            inference_in_queue: None,
            output_processed: None,
        }
    }

    pub fn record_model_proxy_aquired(&mut self) {
        self.model_proxy_aquired = Some(self.start.elapsed())
    }
    pub fn record_serialization_done(&mut self) {
        self.serialization_done = Some(self.start.elapsed())
    }
    pub fn record_inference_in_queue(&mut self) {
        self.inference_in_queue = Some(self.start.elapsed())
    }
    pub fn record_output_processed(&mut self) {
        self.output_processed = Some(self.start.elapsed())
    }

    pub fn record_metrics(&self, model_name: &str, local_metrics: &LocalMetrics) {
        local_metrics
            .observe_model_proxy_aquired(model_name, self.model_proxy_aquired.unwrap().as_secs_f64());
        local_metrics
            .observe_serialization_done(model_name, self.serialization_done.unwrap().as_secs_f64());
        local_metrics
            .observe_inference_in_queue(model_name, self.inference_in_queue.unwrap().as_secs_f64());
        local_metrics
            .observe_output_processed(model_name, self.output_processed.unwrap().as_secs_f64());
    }

    pub fn print_debug(&self) {
        debug!(
            model_proxy_aquired = ?self.model_proxy_aquired,
            deserialization = ?self.serialization_done.unwrap() - self.model_proxy_aquired.unwrap(),
            inference_in_queue = ?self.inference_in_queue.unwrap() - self.serialization_done.unwrap(),
            output_processed = ?self.output_processed.unwrap() - self.inference_in_queue.unwrap(),
            "client trace",
        );
    }
}

pub struct BatchTrace {
    pub batch_first_open: RefCell<std::time::Instant>,
    pub batch_complete: RefCell<std::time::Duration>,
    batch_inference_start: RefCell<std::time::Duration>,
    batch_inference_done: RefCell<std::time::Duration>,
    batch_released: RefCell<std::time::Duration>,
}

impl BatchTrace {
    fn print_debug(&self) {
        debug!(
            batch_complete = ?self.batch_complete.borrow(),
            picked_by_executor = ?*self.batch_inference_start.borrow() - *self.batch_complete.borrow(),
            inference_duration = ?*self.batch_inference_done.borrow() - *self.batch_inference_start.borrow(),
            gather_output = ?*self.batch_released.borrow() - *self.batch_inference_done.borrow(),
            "batch trace",
        );
    }
}
