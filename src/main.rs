#![feature(array_chunks)]
mod grpc;
mod loader;
mod scheduler;
mod tensor;
mod tracing;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tonic::transport::Server;

use ort::{
    execution_providers::{CPUExecutionProvider, OpenVINOExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};
use tokio::sync::RwLock;

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
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    console_subscriber::init();
    let addr = "0.0.0.0:8001".parse()?; // Triton default gRPC port is 8001
                                        // Initialize our service with S3 connection details

    let (input_tx, input_rx) = flume::bounded(4096);

    let mut executor_endpoints = Vec::new();

    let mut model_config: Option<ModelConfig> = None;
    let mut model_metadata: Option<ModelMetadata> = None;

    let batch_size = 16;
    for i in 0..1 {
        let (tx, rx) = flume::bounded(batch_size as usize);
        let vino_provider = OpenVINOExecutionProvider::default()
            .with_device_type("CPU")
            .build();
        let _cpu_provider = CPUExecutionProvider::default().build();
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_execution_providers([vino_provider])
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .with_log_level(ort::logging::LogLevel::Verbose).unwrap()
            .commit_from_file("samples/int64_to_float64.onnx")
            // .commit_from_file("samples/matmul.onnx")
            .unwrap();

        {
            let metadata = session.metadata().unwrap();

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
        }

        let mut executor = loader::OnnxExecutor {
            id: format!("executor-{i}"),
            session,
            batch_size: batch_size as usize,
            inputs: rx,
        };
        tokio::spawn(async move {
            executor.run().await;
        });

        let endpoint = scheduler::ExecutorEndpoint { sender: tx };
        executor_endpoints.push(endpoint);
    }

    let mut sched = scheduler::Scheduler {
        inputs: input_rx,
        executors: executor_endpoints,
        max_queue_time: Duration::from_millis(1),
        batch_size,
    };

    let model_proxy = scheduler::ModelProxy {
        request_sender: input_tx,
        model_config: model_config.unwrap(),
        model_metadata: model_metadata.unwrap(),
    };

    let mut model_map = HashMap::new();
    model_map.insert(String::from("Int64ToFloat64Model"), model_proxy);
    let service = TritonService::new(Arc::from(RwLock::from(model_map)));

    tokio::spawn(async move {
        sched.run().await;
    });

    println!("Starting Triton gRPC server on {}", addr);
    let svc = GrpcInferenceServiceServer::new(service).max_decoding_message_size(128 * 1024 * 1024);
    Server::builder()
        // .layer(
        //     ServiceBuilder::new().layer(TraceLayer::new_for_grpc().make_span_with(
        //         |request: &Request<_>| {
        //             println!("Received request: {:?}", request);
        //             tracing::info_span!("grpc_request")
        //         },
        //     )),
        // )
        .add_service(svc)
        .serve(addr)
        .await?;

    Ok(())
}
