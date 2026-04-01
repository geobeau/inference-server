use std::{collections::HashSet, env, fs, path::PathBuf};

use fusio::{
    fs::{Fs, OpenOptions},
    path::Path as FusioPath,
    remotes::aws::{fs::AmazonS3Builder, AwsCredential},
    Read,
};
use futures::StreamExt;

use super::config::ModelRepositoryConfig;

pub struct LoadedModel {
    pub name: String,
    pub version: u64,
    pub model_path: PathBuf,
    pub config: ModelRepositoryConfig,
}

pub struct ModelRepository {
    fs: fusio::remotes::aws::fs::AmazonS3,
    bucket: String,
    prefix: String,
    local_cache_dir: PathBuf,
}

impl ModelRepository {
    pub fn new(
        endpoint: &str,
        bucket: &str,
        prefix: &str,
        region: &str,
        cache_dir: PathBuf,
    ) -> Self {
        let key_id = env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set");
        let secret_key =
            env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY must be set");

        let fs = AmazonS3Builder::new(bucket.into())
            .credential(AwsCredential {
                key_id,
                secret_key,
                token: None,
            })
            .endpoint(endpoint.into())
            .region(region.into())
            .sign_payload(true)
            .build();

        Self {
            fs,
            bucket: bucket.to_string(),
            prefix: prefix.trim_matches('/').to_string(),
            local_cache_dir: cache_dir,
        }
    }

    pub async fn load_all(
        &self,
        filter: Option<&HashSet<String>>,
    ) -> Result<Vec<LoadedModel>, Box<dyn std::error::Error>> {
        let prefix_path = FusioPath::from(self.prefix.as_str());
        let entries = self.fs.list(&prefix_path).await?;
        futures::pin_mut!(entries);

        // Collect model names from top-level directory entries
        let mut model_names = HashSet::new();
        while let Some(entry) = entries.next().await {
            let entry = entry?;
            let path_str: String = entry.path.into();
            if let Some(rest) = path_str.strip_prefix(&self.prefix) {
                let rest = rest.trim_start_matches('/');
                if let Some(model_name) = rest.split('/').next() {
                    if !model_name.is_empty() {
                        if filter.map_or(true, |f| f.contains(model_name)) {
                            model_names.insert(model_name.to_string());
                        }
                    }
                }
            }
        }

        let mut loaded_models = Vec::new();
        for model_name in model_names {
            let model = self.load_model(&model_name).await?;
            loaded_models.push(model);
        }

        Ok(loaded_models)
    }

    async fn load_model(
        &self,
        model_name: &str,
    ) -> Result<LoadedModel, Box<dyn std::error::Error>> {
        // Download and parse config.yaml
        let config_key = format!("{}/{}/config.yaml", self.prefix, model_name);
        let config_bytes = self.read_file(&config_key).await.map_err(|e| {
            format!(
                "failed to fetch config.yaml for model '{}' at s3://{}/{}: {}",
                model_name, self.bucket, config_key, e
            )
        })?;
        let config: ModelRepositoryConfig = serde_yaml::from_slice(&config_bytes).map_err(|e| {
            format!(
                "failed to parse config.yaml for model '{}': {}",
                model_name, e
            )
        })?;

        // Find the latest version directory
        let model_prefix = format!("{}/{}", self.prefix, model_name);
        let model_prefix_path = FusioPath::from(model_prefix.as_str());
        let entries = self.fs.list(&model_prefix_path).await?;
        futures::pin_mut!(entries);

        let mut latest_version: Option<u64> = None;
        while let Some(entry) = entries.next().await {
            let entry = entry?;
            let path_str: String = entry.path.into();
            if let Some(rest) = path_str.strip_prefix(&model_prefix) {
                let rest = rest.trim_start_matches('/');
                // Look for numeric version directories (e.g. "1/model.onnx")
                if let Some(version_str) = rest.split('/').next() {
                    if let Ok(version) = version_str.parse::<u64>() {
                        if latest_version.map_or(true, |v| version > v) {
                            latest_version = Some(version);
                        }
                    }
                }
            }
        }

        let version = latest_version
            .ok_or_else(|| format!("no version directory found for model '{}'", model_name))?;

        // Download model.onnx
        let onnx_key = format!("{}/{}/{}/model.onnx", self.prefix, model_name, version);
        let onnx_bytes = self.read_file(&onnx_key).await.map_err(|e| {
            format!(
                "failed to fetch model.onnx for model '{}' (version {}) at s3://{}/{}: {}",
                model_name, version, self.bucket, onnx_key, e
            )
        })?;

        // Write to local cache
        let local_dir = self
            .local_cache_dir
            .join(model_name)
            .join(version.to_string());
        std::fs::create_dir_all(&local_dir)?;
        let local_path = local_dir.join("model.onnx");
        std::fs::write(&local_path, &onnx_bytes)?;

        Ok(LoadedModel {
            name: model_name.to_string(),
            version,
            model_path: local_path,
            config,
        })
    }

    async fn read_file(&self, key: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let path = FusioPath::from(key);
        let mut file = self.fs.open_options(&path, OpenOptions::default()).await?;
        let (result, buf) = file.read_to_end_at(Vec::new(), 0).await;
        result?;
        Ok(buf)
    }
}

pub struct LocalModelRepository {
    root: PathBuf,
}

impl LocalModelRepository {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn load_all(&self) -> Result<Vec<LoadedModel>, Box<dyn std::error::Error>> {
        let mut loaded_models = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let model_name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "non-UTF8 model directory name")?;
                let model = self.load_model(&model_name)?;
                loaded_models.push(model);
            }
        }
        Ok(loaded_models)
    }

    fn load_model(&self, model_name: &str) -> Result<LoadedModel, Box<dyn std::error::Error>> {
        let model_dir = self.root.join(model_name);

        // Discover the latest version directory first
        let mut latest_version: Option<u64> = None;
        for entry in fs::read_dir(&model_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Ok(v) = name.parse::<u64>() {
                        if latest_version.map_or(true, |cur| v > cur) {
                            latest_version = Some(v);
                        }
                    }
                }
            }
        }

        let version = latest_version
            .ok_or_else(|| format!("no version directory found for model '{}'", model_name))?;

        let version_dir = model_dir.join(version.to_string());

        // Look for config.yaml in the version directory first, then model root
        let config_path = {
            let version_config = version_dir.join("config.yaml");
            if version_config.exists() {
                version_config
            } else {
                model_dir.join("config.yaml")
            }
        };
        let config_bytes = fs::read(&config_path).map_err(|e| {
            format!(
                "failed to read {}: {e} (each model must contain a config.yaml in its model or version directory)",
                config_path.display()
            )
        })?;
        let config: ModelRepositoryConfig = serde_yaml::from_slice(&config_bytes)?;

        let model_path = version_dir.join("model.onnx");
        if !model_path.exists() {
            return Err(format!("model.onnx not found at {}", model_path.display()).into());
        }

        Ok(LoadedModel {
            name: model_name.to_string(),
            version,
            model_path,
            config,
        })
    }
}
