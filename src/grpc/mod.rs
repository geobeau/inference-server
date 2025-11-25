pub mod compat;
pub mod inference;

use std::ops::Deref;
use std::sync::Arc;
use std::{collections::HashMap, vec};

use arc_swap::ArcSwap;
use inference::grpc_inference_service_server::GrpcInferenceService; // Trait
use inference::*;
use tonic::{Request, Response, Status};

use crate::grpc::inference::model_infer_response::InferOutputTensor;
use crate::scheduler::ModelProxy;
use crate::tensor::batched_tensor::dyntensor_from_bytes;
use crate::tracing::Trace;

#[derive(Clone)]
pub struct TritonService {
    /// Set of model names that are currently considered "loaded"
    pub loaded_models: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>,
}

impl TritonService {
    pub fn new(model_map: Arc<ArcSwap<HashMap<String, Arc<ModelProxy>>>>) -> Self {
        TritonService {
            loaded_models: model_map,
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
        println!("is live?");
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _req: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        println!("is server ready?");
        // Ready if server is up (you could check if models loaded, etc.)
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        println!("is model ready?");
        let _ = request.get_ref().name.clone();
        // For demo, return true if the model is in our loaded set (else false)
        let is_ready = /* check if model_name is loaded */ false;
        Ok(Response::new(ModelReadyResponse { ready: is_ready }))
    }

    async fn server_metadata(
        &self,
        _req: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        println!("server metadata");
        let reply = ServerMetadataResponse {
            name: "inference-server".to_string(),
            version: "1.0.0-demo".to_string(),
            extensions: vec!["classification".to_string(), "model_repository".to_string()],
        };
        Ok(Response::new(reply))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let model_name = &request.get_ref().name;
        println!("Getting model config for {model_name}");
        match self.loaded_models.load().get(model_name) {
            Some(proxy) => Ok(Response::new(ModelMetadataResponse {
                name: model_name.clone(),
                versions: vec![String::from("1")],
                platform: String::from("onnxruntime_onnx"),
                inputs: proxy.model_metadata.input_meta.clone(),
                outputs: proxy.model_metadata.output_meta.clone(),
            })),
            None => Err(Status::not_found(format!("Model {} not found", model_name))),
        }
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let request_ref = request.get_ref();
        let mut trace = Trace::start();

        let models = self.loaded_models.load();

        // Force the lock to be dropped
        let proxy: &ModelProxy = match models.get(&request_ref.model_name) {
            Some(proxy) => proxy,
            None => {
                return Err(Status::not_found(format!(
                    "Model {} not found",
                    &request_ref.model_name
                )))
            }
        };

        let mut inputs = HashMap::new();

        trace.record_serialization_start();
        request_ref
            .inputs
            .iter()
            .enumerate()
            .for_each(|(i, req_input)| {
                let dimensions: Vec<usize> = req_input.shape.iter().map(|i| *i as usize).collect();
                let tensor = dyntensor_from_bytes(
                    DataType::from_str(&req_input.datatype),
                    &dimensions,
                    &request_ref.raw_input_contents[i],
                );
                let input_shape = proxy
                    .model_metadata
                    .input_set
                    .get(&req_input.name)
                    .expect("Input provided not in the model");
                assert_eq!(tensor.shape(), input_shape, "expected the shape to match");
                inputs.insert(req_input.name.clone(), tensor);
            });

        trace.record_dispatch();
        let inference_outputs = proxy.data.infer(&inputs).await.unwrap();
        // println!("resp");
        let mut raw_output: Vec<Vec<u8>> = Vec::new();

        let outputs = inference_outputs
            .iter()
            .map(|(key, output)| {
                let data_type = output.data_type();

                let (shape, serial_data) = output.try_extract_tensor::<f64>().unwrap();
                let bytes: Vec<u8> = serial_data
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                raw_output.push(bytes);

                InferOutputTensor {
                    name: String::from(key),
                    datatype: DataType::from(*data_type).as_str_name().to_string(),
                    shape: Vec::from(shape.deref()),
                    parameters: HashMap::new(),
                    contents: None,
                }
            })
            .collect();

        println!("Request complete");

        Ok(Response::new(ModelInferResponse {
            model_name: proxy.model_config.name.clone(),
            model_version: String::from("1"),
            id: request_ref.id.clone(),
            parameters: HashMap::with_capacity(0),
            outputs,
            raw_output_contents: raw_output,
        }))
    }

    async fn model_config(
        &self,
        request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        let model_name = &request.get_ref().name;
        println!("Getting model config for {model_name}");
        match self.loaded_models.load().get(model_name) {
            Some(proxy) => Ok(Response::new(ModelConfigResponse {
                config: Some(proxy.model_config.clone()),
            })),
            None => Err(Status::not_found(format!("Model {} not found", model_name))),
        }
    }

    async fn model_statistics(
        &self,
        _request: Request<ModelStatisticsRequest>,
    ) -> Result<Response<ModelStatisticsResponse>, Status> {
        println!("stat");
        todo!()
    }

    async fn repository_index(
        &self,
        _request: Request<RepositoryIndexRequest>,
    ) -> Result<Response<RepositoryIndexResponse>, Status> {
        println!("index");
        // Dummy implementation
        Ok(Response::new(RepositoryIndexResponse { models: vec![] }))
    }

    async fn repository_model_load(
        &self,
        _request: Request<RepositoryModelLoadRequest>,
    ) -> Result<Response<RepositoryModelLoadResponse>, Status> {
        println!("load");
        // Dummy implementation
        Ok(Response::new(RepositoryModelLoadResponse {}))
    }

    async fn repository_model_unload(
        &self,
        _request: Request<RepositoryModelUnloadRequest>,
    ) -> Result<Response<RepositoryModelUnloadResponse>, Status> {
        println!("unload");
        // Dummy implementation
        Ok(Response::new(RepositoryModelUnloadResponse {}))
    }

    async fn system_shared_memory_status(
        &self,
        _request: Request<SystemSharedMemoryStatusRequest>,
    ) -> Result<Response<SystemSharedMemoryStatusResponse>, Status> {
        println!("status");
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
