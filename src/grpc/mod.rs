pub mod compat;
pub mod inference;

use std::ops::Deref;
use std::sync::Arc;
use std::{collections::HashMap, vec};

use arc_swap::ArcSwap;
use inference::GrpcInferenceService;
use inference::*;
use pajamax::status::{Code, Status};

use crate::grpc::inference::model_infer_response::InferOutputTensor;
use crate::scheduler::ModelProxy;
use crate::tensor::batched_tensor::dyntensor_from_bytes;
use crate::tracing::ClientTrace;

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

#[async_trait::async_trait(?Send)]
impl GrpcInferenceService for TritonService {
    async fn server_live(&self, _req: ServerLiveRequest) -> Result<ServerLiveResponse, Status> {
        println!("is live?");
        Ok(ServerLiveResponse { live: true })
    }

    async fn server_ready(&self, _req: ServerReadyRequest) -> Result<ServerReadyResponse, Status> {
        println!("is server ready?");
        Ok(ServerReadyResponse { ready: true })
    }

    async fn model_ready(&self, request: ModelReadyRequest) -> Result<ModelReadyResponse, Status> {
        println!("is model ready?");
        let _ = request.name.clone();
        let is_ready = false;
        Ok(ModelReadyResponse { ready: is_ready })
    }

    async fn server_metadata(
        &self,
        _req: ServerMetadataRequest,
    ) -> Result<ServerMetadataResponse, Status> {
        println!("server metadata");
        let reply = ServerMetadataResponse {
            name: "inference-server".to_string(),
            version: "1.0.0-demo".to_string(),
            extensions: vec!["classification".to_string(), "model_repository".to_string()],
        };
        Ok(reply)
    }

    async fn model_metadata(
        &self,
        request: ModelMetadataRequest,
    ) -> Result<ModelMetadataResponse, Status> {
        let model_name = &request.name;
        println!("Getting model config for {model_name}");
        match self.loaded_models.load().get(model_name) {
            Some(proxy) => {
                println!("Got model, responding");
                Ok(ModelMetadataResponse {
                    name: model_name.clone(),
                    versions: vec![String::from("1")],
                    platform: String::from("onnxruntime_onnx"),
                    inputs: proxy.model_metadata.input_meta.clone(),
                    outputs: proxy.model_metadata.output_meta.clone(),
                })
            }
            None => Err(Status {
                code: Code::NotFound,
                message: format!("Model {} not found", model_name),
            }),
        }
    }

    async fn model_infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, Status> {
        let mut trace = ClientTrace::start();

        let models = self.loaded_models.load();

        let proxy: &ModelProxy = match models.get(&request.model_name) {
            Some(proxy) => proxy,
            None => {
                return Err(Status {
                    code: Code::NotFound,
                    message: format!("Model {} not found", &request.model_name),
                })
            }
        };

        let mut inputs = HashMap::new();

        trace.record_model_proxy_aquired();
        request
            .inputs
            .iter()
            .enumerate()
            .for_each(|(i, req_input)| {
                let dimensions: Vec<usize> = req_input.shape.iter().map(|i| *i as usize).collect();
                let tensor = dyntensor_from_bytes(
                    DataType::from_str(&req_input.datatype),
                    &dimensions,
                    &request.raw_input_contents[i],
                );
                let input_shape = proxy
                    .model_metadata
                    .input_set
                    .get(&req_input.name)
                    .expect("Input provided not in the model");
                assert_eq!(tensor.shape(), input_shape, "expected the shape to match");
                inputs.insert(req_input.name.clone(), tensor);
            });

        trace.record_serialization_done();
        let inference_outputs = proxy.data.infer(&inputs, &mut trace).await.unwrap();
        let mut raw_output: Vec<Vec<u8>> = Vec::new();

        let outputs = inference_outputs
            .iter()
            .map(|(key, output)| {
                let data_type = output.data_type();

                let (shape, serial_data) = output.try_extract_tensor::<f32>().unwrap();
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
        trace.record_output_processed();
        trace.print_debug();

        Ok(ModelInferResponse {
            model_name: proxy.model_config.name.clone(),
            model_version: String::from("1"),
            id: request.id.clone(),
            parameters: HashMap::with_capacity(0),
            outputs,
            raw_output_contents: raw_output,
        })
    }

    async fn model_config(
        &self,
        request: ModelConfigRequest,
    ) -> Result<ModelConfigResponse, Status> {
        let model_name = &request.name;
        println!("Getting model config for {model_name}");
        match self.loaded_models.load().get(model_name) {
            Some(proxy) => Ok(ModelConfigResponse {
                config: Some(proxy.model_config.clone()),
            }),
            None => Err(Status {
                code: Code::NotFound,
                message: format!("Model {} not found", model_name),
            }),
        }
    }

    async fn model_statistics(
        &self,
        _request: ModelStatisticsRequest,
    ) -> Result<ModelStatisticsResponse, Status> {
        println!("stat");
        todo!()
    }

    async fn repository_index(
        &self,
        _request: RepositoryIndexRequest,
    ) -> Result<RepositoryIndexResponse, Status> {
        println!("index");
        Ok(RepositoryIndexResponse { models: vec![] })
    }

    async fn repository_model_load(
        &self,
        _request: RepositoryModelLoadRequest,
    ) -> Result<RepositoryModelLoadResponse, Status> {
        println!("load");
        Ok(RepositoryModelLoadResponse {})
    }

    async fn repository_model_unload(
        &self,
        _request: RepositoryModelUnloadRequest,
    ) -> Result<RepositoryModelUnloadResponse, Status> {
        println!("unload");
        Ok(RepositoryModelUnloadResponse {})
    }

    async fn system_shared_memory_status(
        &self,
        _request: SystemSharedMemoryStatusRequest,
    ) -> Result<SystemSharedMemoryStatusResponse, Status> {
        println!("status");
        todo!()
    }

    async fn system_shared_memory_register(
        &self,
        _request: SystemSharedMemoryRegisterRequest,
    ) -> Result<SystemSharedMemoryRegisterResponse, Status> {
        todo!()
    }

    async fn system_shared_memory_unregister(
        &self,
        _request: SystemSharedMemoryUnregisterRequest,
    ) -> Result<SystemSharedMemoryUnregisterResponse, Status> {
        todo!()
    }

    async fn cuda_shared_memory_status(
        &self,
        _request: CudaSharedMemoryStatusRequest,
    ) -> Result<CudaSharedMemoryStatusResponse, Status> {
        todo!()
    }

    async fn cuda_shared_memory_register(
        &self,
        _request: CudaSharedMemoryRegisterRequest,
    ) -> Result<CudaSharedMemoryRegisterResponse, Status> {
        Ok(CudaSharedMemoryRegisterResponse {})
    }

    async fn cuda_shared_memory_unregister(
        &self,
        _request: CudaSharedMemoryUnregisterRequest,
    ) -> Result<CudaSharedMemoryUnregisterResponse, Status> {
        Ok(CudaSharedMemoryUnregisterResponse {})
    }

    async fn trace_setting(
        &self,
        _request: TraceSettingRequest,
    ) -> Result<TraceSettingResponse, Status> {
        todo!()
    }

    async fn log_settings(
        &self,
        _request: LogSettingsRequest,
    ) -> Result<LogSettingsResponse, Status> {
        todo!()
    }
}
