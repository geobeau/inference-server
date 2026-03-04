use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;
use std::sync::atomic::AtomicI64;

pub struct MetricsRegistry {
    registry: Registry,

    pub inference_requests_total: Family<Vec<(String, String)>, Counter>,
    pub inference_batches_total: Family<Vec<(String, String)>, Counter>,

    pub inference_request_duration_seconds: Family<Vec<(String, String)>, Histogram>,
    pub inference_batch_duration_seconds: Family<Vec<(String, String)>, Histogram>,

    pub inference_queue_depth: Family<Vec<(String, String)>, Gauge>,
    pub loaded_models: Gauge<i64, AtomicI64>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let inference_requests_total = Family::<Vec<(String, String)>, Counter>::default();
        registry.register(
            "inference_requests_total",
            "Total number of inference requests",
            inference_requests_total.clone(),
        );

        let inference_batches_total = Family::<Vec<(String, String)>, Counter>::default();
        registry.register(
            "inference_batches_total",
            "Total number of inference batches",
            inference_batches_total.clone(),
        );

        fn make_histogram() -> Histogram {
            Histogram::new(exponential_buckets(0.0001, 2.0, 18))
        }

        let inference_request_duration_seconds =
            Family::<Vec<(String, String)>, Histogram>::new_with_constructor(make_histogram);
        registry.register(
            "inference_request_duration_seconds",
            "Duration of inference requests in seconds",
            inference_request_duration_seconds.clone(),
        );

        let inference_batch_duration_seconds =
            Family::<Vec<(String, String)>, Histogram>::new_with_constructor(make_histogram);
        registry.register(
            "inference_batch_duration_seconds",
            "Duration of inference batches in seconds",
            inference_batch_duration_seconds.clone(),
        );

        let inference_queue_depth = Family::<Vec<(String, String)>, Gauge>::default();
        registry.register(
            "inference_queue_depth",
            "Current depth of the inference queue",
            inference_queue_depth.clone(),
        );

        let loaded_models = Gauge::<i64, AtomicI64>::default();
        registry.register("loaded_models", "Number of currently loaded models", loaded_models.clone());

        Self {
            registry,
            inference_requests_total,
            inference_batches_total,
            inference_request_duration_seconds,
            inference_batch_duration_seconds,
            inference_queue_depth,
            loaded_models,
        }
    }

    pub fn encode(&self) -> String {
        let mut buf = String::new();
        encode(&mut buf, &self.registry).expect("failed to encode metrics");
        buf
    }
}
