use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use prometheus::local::{LocalHistogram, LocalIntCounter};

use super::MetricsRegistry;

const FLUSH_INTERVAL: u64 = 256;

struct LocalModelMetrics {
    requests_ok: LocalIntCounter,
    requests_not_found: LocalIntCounter,
    request_duration: LocalHistogram,
    model_proxy_aquired: LocalHistogram,
    serialization_done: LocalHistogram,
    inference_in_queue: LocalHistogram,
    output_processed: LocalHistogram,
}

pub struct LocalMetrics {
    registry: Arc<MetricsRegistry>,
    per_model: RefCell<HashMap<String, LocalModelMetrics>>,
    ops: Cell<u64>,
}

impl LocalMetrics {
    pub fn new(registry: Arc<MetricsRegistry>) -> Self {
        Self {
            registry,
            per_model: RefCell::new(HashMap::new()),
            ops: Cell::new(0),
        }
    }

    fn maybe_flush(&self) {
        let count = self.ops.get() + 1;
        self.ops.set(count);
        if count % FLUSH_INTERVAL == 0 {
            self.flush();
        }
    }

    fn ensure_model(&self, model: &str) {
        let mut map = self.per_model.borrow_mut();
        if !map.contains_key(model) {
            let r = &self.registry;
            map.insert(
                model.to_string(),
                LocalModelMetrics {
                    requests_ok: r
                        .inference_requests_total
                        .with_label_values(&[model, "ok"])
                        .local(),
                    requests_not_found: r
                        .inference_requests_total
                        .with_label_values(&[model, "not_found"])
                        .local(),
                    request_duration: r
                        .inference_request_duration_seconds
                        .with_label_values(&[model])
                        .local(),
                    model_proxy_aquired: r
                        .inference_requests_model_proxy_aquired
                        .with_label_values(&[model])
                        .local(),
                    serialization_done: r
                        .inference_requests_serialization_done
                        .with_label_values(&[model])
                        .local(),
                    inference_in_queue: r
                        .inference_requests_inference_in_queue
                        .with_label_values(&[model])
                        .local(),
                    output_processed: r
                        .inference_requests_output_processed
                        .with_label_values(&[model])
                        .local(),
                },
            );
        }
    }

    pub fn inc_requests_ok(&self, model: &str) {
        self.ensure_model(model);
        self.per_model.borrow()[model].requests_ok.inc();
        self.maybe_flush();
    }

    pub fn inc_requests_not_found(&self, model: &str) {
        self.ensure_model(model);
        self.per_model.borrow()[model].requests_not_found.inc();
        self.maybe_flush();
    }

    pub fn observe_request_duration(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .request_duration
            .observe(duration);
    }

    pub fn observe_model_proxy_aquired(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .model_proxy_aquired
            .observe(duration);
    }

    pub fn observe_serialization_done(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .serialization_done
            .observe(duration);
    }

    pub fn observe_inference_in_queue(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .inference_in_queue
            .observe(duration);
    }

    pub fn observe_output_processed(&self, model: &str, duration: f64) {
        self.ensure_model(model);
        self.per_model.borrow()[model]
            .output_processed
            .observe(duration);
    }

    pub fn flush(&self) {
        for (_, m) in self.per_model.borrow().iter() {
            m.requests_ok.flush();
            m.requests_not_found.flush();
            m.request_duration.flush();
            m.model_proxy_aquired.flush();
            m.serialization_done.flush();
            m.inference_in_queue.flush();
            m.output_processed.flush();
        }
    }
}

impl Drop for LocalMetrics {
    fn drop(&mut self) {
        self.flush();
    }
}
