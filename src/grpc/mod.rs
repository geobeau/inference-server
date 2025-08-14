pub mod compat;
pub mod inference;

use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;
use std::{collections::HashMap, vec};

use inference::grpc_inference_service_server::GrpcInferenceService; // Trait
use inference::*;
use ort::value::{TensorValueType, Value, ValueType};
use tokio::sync::RwLock;
use tokio::time::Instant;
use tonic::{Request, Response, Status};

use crate::grpc::compat::dyntensor_from_bytes;
use crate::grpc::inference::model_infer_response::InferOutputTensor;
use crate::scheduler::{InferenceRequest, ModelProxy, TracingData};

#[derive(Clone)]
pub struct TritonService {
    /// Set of model names that are currently considered "loaded"
    pub loaded_models: Arc<RwLock<HashMap<String, ModelProxy>>>,
}

impl TritonService {
    pub fn new(model_map: Arc<RwLock<HashMap<String, ModelProxy>>>) -> Self {
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
        let model_name = request.get_ref().name.clone();
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
            name: "ortest".to_string(),
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
        match self.loaded_models.read().await.get(model_name) {
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

    // let mut rng = rand::rng();
    // let input_a = ndarray::Array2::<f32>::from_shape_fn((1024, 1024), |_| rng.random::<f32>());
    // let input_b = ndarray::Array2::<f32>::from_shape_fn((1024, 1024), |_| rng.random::<f32>());

    // let data1 = Tensor::from_array(input_a).unwrap().upcast();
    // let data2 = Tensor::from_array(input_b).unwrap().upcast();

    // let mut data: HashMap<String, DynTensor> = HashMap::with_capacity(2);

    // data.insert(String::from("A"), data1);
    // data.insert(String::from("B"), data2);

    // match input_tx.send(data) {
    //     Ok(_) => println!("success"),
    //     Err(x) => println!("Error {}", x),
    // }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let start: Instant = Instant::now();
        let request_ref = request.get_ref();
        let mut tracing = TracingData {
            start,
            serialization_start: None,
            dispatch: None,
            scheduling_start: None,
            executor_start: None,
            send_response: None,
            process_response: None,
        };

        let proxy: ModelProxy;
        {
            // Force the lock to be dropped
            proxy = match self
                .loaded_models
                .write()
                .await
                .get(&request_ref.model_name)
            {
                Some(proxy) => proxy.clone(),
                None => {
                    return Err(Status::not_found(format!(
                        "Model {} not found",
                        &request_ref.model_name
                    )))
                }
            };
        }

        let (sender, receiver) = flume::bounded(1);
        let mut inputs = HashMap::new();

        tracing.serialization_start = Some(tracing.start.elapsed());
        request_ref
            .inputs
            .iter()
            .enumerate()
            .for_each(|(i, req_input)| {
                // println!(
                //     "{} {} {:?} {}",
                //     req_input.name,
                //     req_input.datatype,
                //     (req_input.shape),
                //     request_ref.raw_input_contents[i].len()
                // );
                let dimensions: Vec<usize> = req_input.shape.iter().map(|i| *i as usize).collect();
                let tensor = dyntensor_from_bytes(
                    DataType::from_str(&req_input.datatype),
                    &dimensions,
                    &request_ref.raw_input_contents[i],
                );

                inputs.insert(req_input.name.clone(), tensor);
            });

        tracing.dispatch = Some(tracing.start.elapsed());
        let req = InferenceRequest {
            inputs,
            resp_chan: sender,
            tracing,
        };
        proxy.request_sender.send_async(req).await.unwrap();
        let mut resp = receiver.recv_async().await.unwrap();

        let mut raw_output: Vec<Vec<u8>> = Vec::new();

        let outputs = resp
            .outputs
            .iter()
            .map(|(key, output)| {
                let data_type = output.data_type();

                let (shape, serial_data) = output.try_extract_tensor::<f64>().unwrap();
                let bytes: Vec<u8> = serial_data
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                raw_output.push(bytes);

                println!("output tensor {:?}", serial_data);

                InferOutputTensor {
                    name: String::from(key),
                    datatype: DataType::from(data_type.clone()).as_str_name().to_string(),
                    shape: Vec::from(shape.deref()),
                    parameters: HashMap::new(),
                    contents: None,
                }
            })
            .collect();

        resp.tracing.process_response = Some(resp.tracing.start.elapsed());
        // println!("{:?}", resp.tracing);

        Ok(Response::new(ModelInferResponse {
            model_name: proxy.model_config.name.clone(),
            model_version: String::from("1"),
            id: request_ref.id.clone(),
            parameters: HashMap::with_capacity(0),
            outputs: outputs,
            raw_output_contents: raw_output,
        }))
    }

    async fn model_config(
        &self,
        request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        let model_name = &request.get_ref().name;
        println!("Getting model config for {model_name}");
        match self.loaded_models.read().await.get(model_name) {
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
