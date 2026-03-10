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
    #[arg(long, default_value_t = 5)]
    pub ort_intra_threads: usize,

    /// ORT inter-op thread pool size (parallelism between independent operators).
    #[arg(long, default_value_t = 5)]
    pub ort_inter_threads: usize,

    // ── Model repository (local or S3, mutually exclusive) ───
    /// Path to a local model repository directory.
    /// Mutually exclusive with --s3-bucket.
    #[arg(long, conflicts_with_all = ["s3_bucket", "s3_endpoint"])]
    pub model_repository: Option<PathBuf>,

    // ── S3 model repository ──────────────────────────────────
    /// S3-compatible endpoint URL (e.g. "https://s3.my-infra.com").
    #[arg(long, requires = "s3_bucket")]
    pub s3_endpoint: Option<String>,

    /// S3 bucket containing the model repository.
    #[arg(long, requires = "s3_endpoint")]
    pub s3_bucket: Option<String>,

    /// Key prefix inside the bucket (e.g. "models/prod").
    #[arg(long, default_value = "")]
    pub s3_prefix: String,

    /// AWS region for the S3 bucket.
    #[arg(long, default_value = "us-east-1")]
    pub s3_region: String,

    /// Local directory to cache downloaded S3 models.
    #[arg(long, default_value = "/tmp/model_cache")]
    pub model_cache_dir: PathBuf,

    /// Comma-separated list of model names to load at startup.
    /// If omitted, all discovered models are loaded.
    #[arg(long, value_delimiter = ',')]
    pub load_models: Option<Vec<String>>,
}

pub enum ModelSource {
    Local(PathBuf),
    S3 {
        endpoint: String,
        bucket: String,
        prefix: String,
        region: String,
        cache_dir: PathBuf,
    },
}

impl Args {
    pub fn model_source(&self) -> ModelSource {
        if let Some(ref path) = self.model_repository {
            ModelSource::Local(path.clone())
        } else if let (Some(ref endpoint), Some(ref bucket)) =
            (&self.s3_endpoint, &self.s3_bucket)
        {
            ModelSource::S3 {
                endpoint: endpoint.clone(),
                bucket: bucket.clone(),
                prefix: self.s3_prefix.clone(),
                region: self.s3_region.clone(),
                cache_dir: self.model_cache_dir.clone(),
            }
        } else {
            eprintln!("error: either --model-repository or --s3-bucket + --s3-endpoint is required");
            std::process::exit(1);
        }
    }
}

fn default_num_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
