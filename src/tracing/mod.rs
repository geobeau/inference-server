use std::{cell::RefCell, time::Duration};

use tokio::time::Instant;

#[derive(Debug)]
pub struct ClientTrace {
    pub start: Instant,
    model_proxy_aquired: Option<Duration>,
    serialization_done: Option<Duration>,
    inference_in_queue: Option<Duration>,
    output_processed: Option<Duration>,
}

impl ClientTrace {
    pub fn start() -> ClientTrace {
        let start: Instant = Instant::now();
        ClientTrace {
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
}

impl ClientTrace {
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
