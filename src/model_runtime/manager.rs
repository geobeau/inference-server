use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::grpc::inference::{
    model_metadata_response::TensorMetadata, DataType, ModelConfig, ModelInput, ModelOutput,
};
use crate::metrics::MetricsRegistry;
use crate::model_repository::config::{AllocatorKind, ModelRepositoryConfig};
use crate::scheduler::{ModelInputMetadata, ModelMetadata, ModelProxy};
use crate::tensor::supertensor::SuperTensorBuffer;

use super::session_starter::SessionStartRequest;

pub struct LoadModelRequest {
    pub model_name: String,
    pub version: u64,
    pub model_path: PathBuf,
    pub config: ModelRepositoryConfig,
    pub reply: oneshot::Sender<Result<(), LoadError>>,
}

#[derive(Debug)]
pub enum LoadError {
    SessionBuild(String),
    AllocatorCreate(String),
    SuperTensorBuild,
    DispatchFailed(String),
}

pub struct ModelRuntimeManager {
    receiver: mpsc::Receiver<LoadModelRequest>,
    starters: Vec<mpsc::Sender<SessionStartRequest>>,
    loaded_models: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>,
    metrics: Arc<MetricsRegistry>,
    round_robin: AtomicUsize,
}

impl ModelRuntimeManager {
    pub fn new(
        receiver: mpsc::Receiver<LoadModelRequest>,
        starters: Vec<mpsc::Sender<SessionStartRequest>>,
        loaded_models: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>,
        metrics: Arc<MetricsRegistry>,
    ) -> Self {
        Self {
            receiver,
            starters,
            loaded_models,
            metrics,
            round_robin: AtomicUsize::new(0),
        }
    }

    pub async fn run(mut self) {
        while let Some(req) = self.receiver.recv().await {
            let result = self
                .load_model(req.model_name, req.model_path, &req.config)
                .await;
            let _ = req.reply.send(result);
        }
    }

    async fn load_model(
        &self,
        model_name: String,
        model_path: PathBuf,
        config: &ModelRepositoryConfig,
    ) -> Result<(), LoadError> {
        println!("Loading model {model_name} at {model_path:?}");
        let num_executors = config.num_executors;
        let batch_size = config.batch_size;
        let capacity = config.capacity;

        // Build N sessions
        let mut sessions = Vec::with_capacity(num_executors);
        for _ in 0..num_executors {
            let session = Session::builder()
                .map_err(|e| LoadError::SessionBuild(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| LoadError::SessionBuild(e.to_string()))?
                .commit_from_file(&model_path)
                .map_err(|e| LoadError::SessionBuild(e.to_string()))?;
            sessions.push(session);
        }

        // Build SuperTensorBuffer and metadata from the first session
        let first_session = &sessions[0];

        let alloc_device = match config.allocator {
            AllocatorKind::Cpu => AllocationDevice::CPU,
            AllocatorKind::CudaPinned => AllocationDevice::CUDA_PINNED,
        };
        let memory_type = match config.allocator {
            AllocatorKind::Cpu => MemoryType::CPUInput,
            AllocatorKind::CudaPinned => MemoryType::CPUInput,
        };

        let allocator = Allocator::new(
            first_session,
            MemoryInfo::new(alloc_device, 0, AllocatorType::Device, memory_type)
                .map_err(|e| LoadError::AllocatorCreate(e.to_string()))?,
        )
        .map_err(|e| LoadError::AllocatorCreate(e.to_string()))?;

        let inputs_ref: Vec<_> = first_session.inputs().iter().collect();
        let super_tensor_buffer =
            SuperTensorBuffer::new(capacity, batch_size, &inputs_ref, &allocator)
                .map_err(|_| LoadError::SuperTensorBuild)?;

        let metadata_name = first_session
            .metadata()
            .map_err(|e| LoadError::SessionBuild(e.to_string()))?
            .name()
            .unwrap_or(model_name.clone());

        let inputs = first_session
            .inputs()
            .iter()
            .map(|input| {
                let tensor_type: &DataType = &input.dtype().tensor_type().unwrap().into();
                let tensor_shape: Vec<i64> = input
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .skip_while(|(i, dim)| *i == 0 && **dim == -1 && batch_size > 0)
                    .map(|(_, dim)| dim)
                    .cloned()
                    .collect();
                ModelInput {
                    name: input.name().to_string(),
                    data_type: (*tensor_type).into(),
                    format: 0,
                    dims: tensor_shape,
                    reshape: None,
                    is_shape_tensor: false,
                    allow_ragged_batch: false,
                    optional: false,
                    is_non_linear_format_io: false,
                }
            })
            .collect();

        let outputs = first_session
            .outputs()
            .iter()
            .map(|output| {
                let tensor_type: &DataType = &output.dtype().tensor_type().unwrap().into();
                let tensor_shape: Vec<i64> = output
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .skip_while(|(i, dim)| *i == 0 && **dim == -1)
                    .map(|(_, dim)| dim)
                    .cloned()
                    .collect();
                ModelOutput {
                    name: output.name().to_string(),
                    data_type: (*tensor_type).into(),
                    dims: tensor_shape,
                    reshape: None,
                    is_shape_tensor: false,
                    is_non_linear_format_io: false,
                    label_filename: String::from(""),
                }
            })
            .collect();

        let mut input_set = HashMap::new();
        let inputs_metadata = first_session
            .inputs()
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let tensor_type: &DataType = &input.dtype().tensor_type().unwrap().into();
                let tensor_shape: Vec<i64> = input
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();
                let mut input_shape = input.dtype().tensor_shape().unwrap().clone();
                input_shape[0] = 1;
                let input_metadata = ModelInputMetadata {
                    shape: input_shape,
                    order: i,
                };
                input_set.insert(input.name().to_string(), input_metadata);
                TensorMetadata {
                    name: input.name().to_string(),
                    datatype: tensor_type.to_metadata_string(),
                    shape: tensor_shape,
                }
            })
            .collect();

        let outputs_metadata = first_session
            .outputs()
            .iter()
            .map(|output| {
                let tensor_type: &DataType = &output.dtype().tensor_type().unwrap().into();
                let tensor_shape: Vec<i64> = output
                    .dtype()
                    .tensor_shape()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();
                TensorMetadata {
                    name: output.name().to_string(),
                    datatype: tensor_type.to_metadata_string(),
                    shape: tensor_shape,
                }
            })
            .collect();

        let model_metadata = ModelMetadata {
            input_meta: inputs_metadata,
            output_meta: outputs_metadata,
            input_set,
        };

        let model_config = ModelConfig {
            name: metadata_name,
            platform: String::from("onnxruntime_onnx"),
            backend: String::from("onnxruntime"),
            runtime: String::from("onnxruntime"),
            version_policy: None,
            max_batch_size: batch_size as i32,
            input: inputs,
            output: outputs,
            batch_input: vec![],
            batch_output: vec![],
            optimization: None,
            instance_group: vec![],
            default_model_filename: String::from("todo"),
            cc_model_filenames: HashMap::new(),
            metric_tags: HashMap::new(),
            parameters: HashMap::new(),
            model_warmup: vec![],
            model_operations: None,
            model_transaction_policy: None,
            model_repository_agents: None,
            response_cache: None,
            model_metrics: None,
            scheduling_choice: None,
        };

        let model_proxy = Arc::new(ModelProxy {
            data: super_tensor_buffer,
            model_config,
            model_metadata,
        });

        // Register in the shared model map
        let current = self.loaded_models.load();
        let mut new_map = (**current).clone();
        new_map.insert(model_name, model_proxy.clone());
        self.loaded_models.store(Arc::new(new_map));
        self.metrics.loaded_models.inc();

        // Dispatch sessions round-robin to session starters
        for (i, session) in sessions.into_iter().enumerate() {
            let idx = self.round_robin.fetch_add(1, Ordering::Relaxed) % self.starters.len();
            self.starters[idx]
                .send(SessionStartRequest {
                    executor_id: format!("executor-{i}"),
                    session,
                    model_proxy: model_proxy.clone(),
                })
                .await
                .map_err(|e| LoadError::DispatchFailed(e.to_string()))?;
        }

        Ok(())
    }
}
