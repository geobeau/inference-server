mod grpc;
mod loader;
mod scheduler;
mod tensor;
mod tracing;
use arc_swap::ArcSwap;
use log::info;
use std::{collections::HashMap, sync::Arc};
use tonic::transport::Server;
use tower::limit::ConcurrencyLimitLayer;

use ort::{
    environment::{EnvironmentBuilder, GlobalThreadPoolOptions}, execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, OpenVINOExecutionProvider}, session::{Session, builder::GraphOptimizationLevel}
};

use crate::{
    grpc::{
        inference::{
            grpc_inference_service_server::GrpcInferenceServiceServer,
            model_metadata_response::TensorMetadata, DataType, ModelConfig, ModelInput,
            ModelOutput,
        },
        TritonService,
    },
    scheduler::ModelMetadata,
    tensor::{supertensor::SuperTensorBuffer, tensor_ringbuffer::BatchRingBuffer},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // console_subscriber::init();
    let addr = "0.0.0.0:8001".parse()?; // Triton default gRPC port is 8001
                                        // Initialize our service with S3 connection details

    let mut model_config: Option<ModelConfig> = None;
    let mut model_metadata: Option<ModelMetadata> = None;

    let mut super_tensor_buffer: Option<SuperTensorBuffer> = None;
    let mut model_proxy: Option<Arc<scheduler::ModelProxy>> = None;

    let batch_size = 64;
    let capacity = 64;

    let cuda_provider = CUDAExecutionProvider::default().build().error_on_failure();
    let thread_pool =  GlobalThreadPoolOptions::default().with_intra_threads(5).unwrap().with_spin_control(true).unwrap();
    ort::init().with_execution_providers([cuda_provider]).with_global_thread_pool(thread_pool).commit().unwrap();
    // Cannot be more than the ring buffer size
    for i in 0..8 {
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file("samples/model.onnx")
            // .commit_from_file("samples/matmul.onnx")
            .unwrap();

        if i == 0 {
            let metadata = session.metadata().unwrap();
            let inputs = session.inputs.iter().collect();
            super_tensor_buffer =
                Some(SuperTensorBuffer::new(capacity, batch_size as usize, &inputs).unwrap());

            let inputs = session
                .inputs
                .iter()
                .map(|input| {
                    let tensor_type: &DataType = &input.input_type.tensor_type().unwrap().into();
                    let tensor_shape: Vec<i64> = input
                        .input_type
                        .tensor_shape()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .skip_while(|(i, dim)| *i == 0 && **dim == -1 && batch_size > 0)
                        .map(|(_, dim)| dim)
                        .cloned()
                        .collect();
                    // println!("tensor dims {} {:?}", input.name.clone(), tensor_shape);
                    // let tensor_ = input.input_type
                    ModelInput {
                        name: input.name.clone(),
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

            let outputs = session
                .outputs
                .iter()
                .map(|output| {
                    let tensor_type: &DataType = &output.output_type.tensor_type().unwrap().into();
                    let tensor_shape: Vec<i64> = output
                        .output_type
                        .tensor_shape()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .skip_while(|(i, dim)| *i == 0 && **dim == -1)
                        .map(|(_, dim)| dim)
                        .cloned()
                        .collect();
                    // println!("tensor dims {} {:?}", input.name.clone(), tensor_shape);
                    // let tensor_ = input.input_type
                    ModelOutput {
                        name: output.name.clone(),
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
            let input_metadata = session
                .inputs
                .iter()
                .map(|input| {
                    let tensor_type: &DataType = &input.input_type.tensor_type().unwrap().into();
                    let tensor_shape: Vec<i64> = input
                        .input_type
                        .tensor_shape()
                        .unwrap()
                        .iter()
                        .cloned()
                        .collect();
                    // println!("tensor shape {} {:?}", input.name.clone(), tensor_shape);
                    // let tensor_ = input.input_type
                    let mut input_shape = input.input_type.tensor_shape().unwrap().clone();
                    input_shape[0] = 1;
                    input_set.insert(input.name.clone(), input_shape);
                    TensorMetadata {
                        name: input.name.clone(),
                        datatype: tensor_type.to_metadata_string(),
                        shape: tensor_shape,
                    }
                })
                .collect();

            let output_metadata = session
                .outputs
                .iter()
                .map(|output| {
                    let tensor_type: &DataType = &output.output_type.tensor_type().unwrap().into();
                    let tensor_shape: Vec<i64> = output
                        .output_type
                        .tensor_shape()
                        .unwrap()
                        .iter()
                        .cloned()
                        .collect();
                    // println!("tensor shape {} {:?}", input.name.clone(), tensor_shape);
                    // let tensor_ = input.input_type
                    TensorMetadata {
                        name: output.name.clone(),
                        datatype: tensor_type.to_metadata_string(),
                        shape: tensor_shape,
                    }
                })
                .collect();

            model_metadata = Some(ModelMetadata {
                input_meta: input_metadata,
                output_meta: output_metadata,
                input_set,
            });

            model_config = Some(ModelConfig {
                name: metadata.name().unwrap(),
                platform: String::from("onnxruntime_onnx"),
                backend: String::from("onnxruntime"),
                runtime: String::from("onnxruntime"),
                version_policy: None,
                max_batch_size: batch_size,
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
            });

            model_proxy = Some(Arc::from(scheduler::ModelProxy {
                data: super_tensor_buffer.unwrap(),
                model_config: model_config.unwrap(),
                model_metadata: model_metadata.unwrap(),
            }));
        }

        let mut executor = loader::OnnxExecutor {
            id: format!("executor-{i}"),
            session,
            model: model_proxy.clone().unwrap(),
        };
        tokio::spawn(async move {
            executor.run().await;
        });
    }

    let mut model_map = HashMap::new();
    model_map.insert(
        String::from("Int64ToFloat64Model"),
        model_proxy.clone().unwrap(),
    );
    let service = TritonService::new(Arc::from(ArcSwap::from_pointee(model_map)));

    info!("Starting Triton gRPC server on {}", addr);
    let svc = GrpcInferenceServiceServer::new(service).max_decoding_message_size(128 * 1024 * 1024);
    let layer = ConcurrencyLimitLayer::new(64 * 64);

    Server::builder()
        .layer(layer)
        .add_service(svc)
        .serve(addr)
        .await?;

    Ok(())
}
