use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(name = "inference-server")]
pub struct Args {
    // ── Thread pool ──────────────────────────────────────────
    /// Number of compio worker threads (one per core).
    /// Defaults to the number of available CPUs.
    #[arg(long, default_value_t = default_num_cores())]
    pub num_cores: usize,

    /// ORT intra-op thread pool size (parallelism within a single operator).
    #[arg(long, default_value_t = 1)]
    pub ort_intra_threads: usize,

    /// ORT inter-op thread pool size (parallelism between independent operators).
    #[arg(long, default_value_t = 1)]
    pub ort_inter_threads: usize,

    // ── S3 model repository ──────────────────────────────────
    /// S3-compatible endpoint URL (e.g. "https://s3.my-infra.com").
    #[arg(long)]
    pub s3_endpoint: String,

    /// S3 bucket containing the model repository.
    #[arg(long)]
    pub s3_bucket: String,

    /// Key prefix inside the bucket (e.g. "models/prod").
    #[arg(long, default_value = "")]
    pub s3_prefix: String,

    /// AWS region for the S3 bucket.
    #[arg(long, default_value = "us-east-1")]
    pub s3_region: String,

    /// Local directory to cache downloaded models.
    #[arg(long, default_value = "/tmp/model_cache")]
    pub model_cache_dir: PathBuf,
}

fn default_num_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
