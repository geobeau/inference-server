use std::cell::RefCell;
use std::time::Duration;

use tokio::time::Instant;

use crate::metrics::LocalMetrics;

pub struct ClientTrace<'a> {
    local_metrics: &'a LocalMetrics,
    pub start: Instant,
    model_proxy_aquired: Option<Duration>,
    serialization_done: Option<Duration>,
    inference_in_queue: Option<Duration>,
    output_processed: Option<Duration>,
}

impl<'a> ClientTrace<'a> {
    pub fn start(local_metrics: &'a LocalMetrics) -> ClientTrace<'a> {
        let start: Instant = Instant::now();
        ClientTrace {
            local_metrics,
            start,
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

    pub fn record_metrics(&self, model_name: &str) {
        self.local_metrics
            .observe_model_proxy_aquired(model_name, self.model_proxy_aquired.unwrap().as_secs_f64());
        self.local_metrics
            .observe_serialization_done(model_name, self.serialization_done.unwrap().as_secs_f64());
        self.local_metrics
            .observe_inference_in_queue(model_name, self.inference_in_queue.unwrap().as_secs_f64());
        self.local_metrics
            .observe_output_processed(model_name, self.output_processed.unwrap().as_secs_f64());
    }

    pub fn print_debug(&self) {
        println!("---\ntime to aquire model config {:?}\n deserialize the proto {:?}\n inference in queue {:?}\n output is processed {:?}", 
            self.model_proxy_aquired,
            self.serialization_done.unwrap() - self.model_proxy_aquired.unwrap(),
            self.inference_in_queue.unwrap() - self.serialization_done.unwrap(),
            self.output_processed.unwrap() - self.inference_in_queue.unwrap(),
        )
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
        println!("---\ntime to complete batch {:?}\n picked by executor {:?}\n inference duration {:?}\n time to gather output {:?}", 
            self.batch_complete.borrow(),
            *self.batch_inference_start.borrow() - *self.batch_complete.borrow(),
            *self.batch_inference_done.borrow() - *self.batch_inference_start.borrow(),
            *self.batch_released.borrow() - *self.batch_inference_done.borrow(),
        )
    }
}
