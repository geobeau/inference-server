mod grpc;
mod loader;
mod scheduler;
mod tensor;
mod tracing;
use arc_swap::ArcSwap;
use log::info;
use pajamax::{Server, serve};
use std::{collections::HashMap, sync::Arc};

use ort::{
    environment::{EnvironmentBuilder, GlobalThreadPoolOptions},
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, OpenVINOExecutionProvider},
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::{
    grpc::{
        TritonService, inference::{
            DataType, GrpcInferenceServiceServer, ModelConfig, ModelInput, ModelOutput, model_metadata_response::TensorMetadata
        }
    },
    scheduler::{ModelInputMetadata, ModelMetadata},
    tensor::supertensor::SuperTensorBuffer,
};

// Current worker that I use is 16 vcpu: 12 is for compio and 4 are dedicated to onnx (see with_intra_threads, minus 1)
// TODO: make this configurable
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_cores = 2;
    let num_executors = 8;
    let batch_size = 64;
    let capacity = 64;

    let cuda_provider = CUDAExecutionProvider::default()
        .with_device_id(0)
        .build()
        .error_on_failure();
    let thread_pool = GlobalThreadPoolOptions::default()
        .with_intra_threads(4)
        .unwrap()
        .with_inter_threads(4).unwrap()
        .with_spin_control(false)
        .unwrap();
    ort::init()
        .with_execution_providers([cuda_provider])
        .with_global_thread_pool(thread_pool)
        .commit();

    // Create all sessions and extract metadata from the first one
    let mut sessions = Vec::with_capacity(num_executors);
    let mut model_proxy: Option<Arc<scheduler::ModelProxy>> = None;

    for i in 0..num_executors {
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file("samples/model.onnx")
            .unwrap();

        if i == 0 {
            let metadata = session.metadata().unwrap();
            let allocator = Allocator::new(
                &session,
                MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    0,
                    AllocatorType::Device,
                    MemoryType::CPUInput,
                )?,
            )?;
            let inputs = session.inputs().iter().collect();
            let super_tensor_buffer =
                SuperTensorBuffer::new(capacity, batch_size as usize, &inputs, &allocator).unwrap();

            let inputs = session
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

            let outputs = session
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
            let inputs_metadata = session
                .inputs()
                .iter()
                .enumerate()
                .map(|(usize, input)| {
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
                    println!("{:?} -> ({:?},{:?})", input.name(), tensor_type, tensor_shape);
                    let input_metadata = ModelInputMetadata { shape: input_shape, order: i };
                    input_set.insert(input.name().to_string(), input_metadata);
                    TensorMetadata {
                        name: input.name().to_string(),
                        datatype: tensor_type.to_metadata_string(),
                        shape: tensor_shape,
                    }
                })
                .collect();

            let outputs_metadata = session
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
            };

            model_proxy = Some(Arc::from(scheduler::ModelProxy {
                data: super_tensor_buffer,
                model_config,
                model_metadata,
            }));
        }

        sessions.push(session);
    }

    let model_proxy = model_proxy.unwrap();

    let mut model_map = HashMap::new();
    model_map.insert(String::from("Int64ToFloat64Model"), model_proxy.clone());
    let loaded_models = Arc::new(ArcSwap::from_pointee(model_map));

    // Distribute executors to cores round-robin
    let mut per_core_sessions: Vec<Vec<(usize, Session)>> =
        (0..num_cores).map(|_| Vec::new()).collect();
    for (i, session) in sessions.into_iter().enumerate() {
        per_core_sessions[i % num_cores].push((i, session));
    }

    let addr = "0.0.0.0:8001";
    info!("Starting Triton gRPC server on {}", addr);

    let config = pajamax::Config::new()
        .max_concurrent_connections(100000)
        .max_concurrent_streams(100000)
        .max_frame_size(32 * 1024);

    // Spawn one thread per core, each running executors + pajamax listener on the same compio runtime
    let mut handles = Vec::new();

    let services: Vec<Box<dyn Fn() -> std::rc::Rc<dyn pajamax::PajamaxService> + Send + Sync>> =
        vec![Box::new(move || {
            std::rc::Rc::new(GrpcInferenceServiceServer::new(TritonService::new(
                loaded_models.clone(),
            )))
        })];
    let grpc_server = Server::new(services, config, addr.to_string());

    for (core_id, core_sessions) in per_core_sessions.into_iter().enumerate() {
        let model_proxy = model_proxy.clone();
        let server = grpc_server.clone();
        let handle = std::thread::Builder::new()
            .name(format!("core-{core_id}"))
            .spawn(move || {
                
                let mut proactor = compio::driver::ProactorBuilder::new();
                // configs taken from apache iggy
                proactor
                    .capacity(4096)
                    .coop_taskrun(true)
                    .taskrun_flag(true);

                let rt = compio::runtime::RuntimeBuilder::new()
                    .with_proactor(proactor.to_owned())
                    .event_interval(128)
                    .build()
                    .expect("failed to build compio runtime");

                rt.block_on(async move {
                    // Spawn executors assigned to this core
                    for (i, session) in core_sessions {
                        let model_proxy = model_proxy.clone();
                        compio::runtime::spawn(async move {
                            let mut executor = loader::OnnxExecutor {
                                id: format!("executor-{i}"),
                                session,
                                model: model_proxy,
                            };
                            executor.run().await;
                        })
                        .detach();
                    }

                    serve(server).await
                }).unwrap();
            })
            .unwrap();
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }

    Ok(())
}
