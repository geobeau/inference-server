use std::{collections::HashSet, env, path::PathBuf};

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
    pub model_path: PathBuf,
    pub config: ModelRepositoryConfig,
}

pub struct ModelRepository {
    fs: fusio::remotes::aws::fs::AmazonS3,
    prefix: String,
    local_cache_dir: PathBuf,
}

impl ModelRepository {
    pub fn new(bucket: &str, prefix: &str, region: &str, cache_dir: PathBuf) -> Self {
        let key_id = env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID must be set");
        let secret_key =
            env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY must be set");

        let fs = AmazonS3Builder::new(bucket.into())
            .credential(AwsCredential {
                key_id,
                secret_key,
                token: None,
            })
            .region(region.into())
            .sign_payload(true)
            .build();

        Self {
            fs,
            prefix: prefix.trim_matches('/').to_string(),
            local_cache_dir: cache_dir,
        }
    }

    pub async fn load_all(&self) -> Result<Vec<LoadedModel>, Box<dyn std::error::Error>> {
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
                        model_names.insert(model_name.to_string());
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
        let config_bytes = self.read_file(&config_key).await?;
        let config: ModelRepositoryConfig = serde_yaml::from_slice(&config_bytes)?;

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
        let onnx_bytes = self.read_file(&onnx_key).await?;

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
