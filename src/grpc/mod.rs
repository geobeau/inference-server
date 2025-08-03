pub mod inference;

use std::collections::HashMap;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use inference::grpc_inference_service_server::GrpcInferenceService; // Trait
use inference::*;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct TritonService {
    /// Set of model names that are currently considered "loaded"
    pub loaded_models: Arc<RwLock<HashMap<String, String>>>,
}

impl TritonService {
    pub fn new() -> Self {
        TritonService {
            loaded_models: Arc::from(RwLock::from(HashMap::new())),
        }
    }
}

unsafe impl Send for ModelInferResponse {}

#[tonic::async_trait]
impl GrpcInferenceService for TritonService {
    async fn server_live(
        &self,
        _req: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        // Always live for demonstration
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _req: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        // Ready if server is up (you could check if models loaded, etc.)
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let model_name = request.get_ref().name.clone();
        // For demo, return true if the model is in our loaded set (else false)
        let is_ready = /* check if model_name is loaded */ false;
        Ok(Response::new(ModelReadyResponse { ready: is_ready }))
    }

    async fn server_metadata(
        &self,
        _req: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        // Return some static metadata
        let reply = ServerMetadataResponse {
            name: "TritonRustServer".to_string(),
            version: "1.0.0-demo".to_string(),
            extensions: vec!["classification".to_string(), "model_repository".to_string()],
        };
        Ok(Response::new(reply))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let model_name = request.get_ref().name.clone();
        // If model is loaded, return dummy metadata; otherwise error
        if
        /* model loaded */
        false {
            let resp = ModelMetadataResponse {
                name: model_name,
                versions: vec!["1".to_string()],
                platform: "onnxruntime".to_string(),
                inputs: vec![],
                outputs: vec![],
            };
            Ok(Response::new(resp))
        } else {
            Err(Status::not_found(format!("Model {} not found", model_name)))
        }
    }

    async fn model_infer(
        &self,
        _req: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        todo!()
    }

    async fn model_config(
        &self,
        _request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        todo!()
    }

    async fn model_statistics(
        &self,
        _request: Request<ModelStatisticsRequest>,
    ) -> Result<Response<ModelStatisticsResponse>, Status> {
        todo!()
    }

    async fn repository_index(
        &self,
        _request: Request<RepositoryIndexRequest>,
    ) -> Result<Response<RepositoryIndexResponse>, Status> {
        // Dummy implementation
        Ok(Response::new(RepositoryIndexResponse { models: vec![] }))
    }

    async fn repository_model_load(
        &self,
        _request: Request<RepositoryModelLoadRequest>,
    ) -> Result<Response<RepositoryModelLoadResponse>, Status> {
        // Dummy implementation
        Ok(Response::new(RepositoryModelLoadResponse {}))
    }

    async fn repository_model_unload(
        &self,
        _request: Request<RepositoryModelUnloadRequest>,
    ) -> Result<Response<RepositoryModelUnloadResponse>, Status> {
        // Dummy implementation
        Ok(Response::new(RepositoryModelUnloadResponse {}))
    }

    async fn system_shared_memory_status(
        &self,
        _request: Request<SystemSharedMemoryStatusRequest>,
    ) -> Result<Response<SystemSharedMemoryStatusResponse>, Status> {
        todo!()
    }

    async fn system_shared_memory_register(
        &self,
        _request: Request<SystemSharedMemoryRegisterRequest>,
    ) -> Result<Response<SystemSharedMemoryRegisterResponse>, Status> {
        todo!()
    }

    async fn system_shared_memory_unregister(
        &self,
        _request: Request<SystemSharedMemoryUnregisterRequest>,
    ) -> Result<Response<SystemSharedMemoryUnregisterResponse>, Status> {
        todo!()
    }

    async fn cuda_shared_memory_status(
        &self,
        _request: Request<CudaSharedMemoryStatusRequest>,
    ) -> Result<Response<CudaSharedMemoryStatusResponse>, Status> {
        // Dummy implementation
        todo!()
    }

    async fn cuda_shared_memory_register(
        &self,
        _request: Request<CudaSharedMemoryRegisterRequest>,
    ) -> Result<Response<CudaSharedMemoryRegisterResponse>, Status> {
        // Dummy implementation
        Ok(Response::new(CudaSharedMemoryRegisterResponse {}))
    }

    async fn cuda_shared_memory_unregister(
        &self,
        _request: Request<CudaSharedMemoryUnregisterRequest>,
    ) -> Result<Response<CudaSharedMemoryUnregisterResponse>, Status> {
        // Dummy implementation
        Ok(Response::new(CudaSharedMemoryUnregisterResponse {}))
    }

    async fn trace_setting(
        &self,
        _request: Request<TraceSettingRequest>,
    ) -> Result<Response<TraceSettingResponse>, Status> {
        todo!()
    }

    async fn log_settings(
        &self,
        _request: Request<LogSettingsRequest>,
    ) -> Result<Response<LogSettingsResponse>, Status> {
        // Dummy implementation
        todo!()
    }
}
