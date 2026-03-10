mod grpc;
mod loader;
mod metrics;
mod model_repository;
mod model_runtime;
mod scheduler;
mod tensor;
mod tracing;
use arc_swap::ArcSwap;
use pajamax::{serve, Server};
use ::tracing::info;
use tracing_subscriber::EnvFilter;
use std::path::PathBuf;
use std::{collections::HashMap, sync::Arc};

use ort::{environment::GlobalThreadPoolOptions, execution_providers::CUDAExecutionProvider};

use crate::{
    grpc::{inference::GrpcInferenceServiceServer, TritonService},
    model_repository::config::{AllocatorKind, Backend, ModelRepositoryConfig},
    model_runtime::{LoadModelRequest, ModelRuntimeManager, SessionStarter},
};

// Current worker that I use is 16 vcpu: 12 is for compio and 4 are dedicated to onnx (see with_intra_threads, minus 1)
// TODO: make this configurable
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_file(true)
        .with_line_number(true)
        .with_target(true)
        .init();

    let num_cores = 16;

    let cuda_provider = CUDAExecutionProvider::default()
        .with_device_id(0)
        .build()
        .error_on_failure();
    let thread_pool = GlobalThreadPoolOptions::default()
        .with_intra_threads(7)
        .unwrap()
        .with_inter_threads(4)
        .unwrap()
        .with_spin_control(false)
        .unwrap();
    ort::init()
        .with_execution_providers([cuda_provider])
        .with_global_thread_pool(thread_pool)
        .commit();

    // Metrics registry
    let metrics_registry = Arc::new(metrics::MetricsRegistry::new());

    // Shared model map for gRPC handlers
    let loaded_models: Arc<ArcSwap<HashMap<String, Arc<scheduler::ModelProxy>>>> =
        Arc::new(ArcSwap::from_pointee(HashMap::new()));

    // Create the load-model channel
    let (load_tx, load_rx) = tokio::sync::mpsc::channel::<LoadModelRequest>(16);

    // Create per-core session starter channels
    let mut starter_txs = Vec::with_capacity(num_cores);
    let mut starter_rxs = Vec::with_capacity(num_cores);
    for _ in 0..num_cores {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        starter_txs.push(tx);
        starter_rxs.push(Some(rx));
    }

    let addr = "0.0.0.0:8001";
    info!("Starting Triton gRPC server on {}", addr);

    let config = pajamax::Config::new()
        .max_concurrent_connections(100000)
        .max_concurrent_streams(100000)
        .max_frame_size(32 * 1024);

    let grpc_loaded_models = loaded_models.clone();
    let services: Vec<Box<dyn Fn() -> std::rc::Rc<dyn pajamax::PajamaxService> + Send + Sync>> =
        vec![Box::new(move || {
            std::rc::Rc::new(GrpcInferenceServiceServer::new(TritonService::new(
                grpc_loaded_models.clone(),
            )))
        })];
    let grpc_server = Server::new(services, config, addr.to_string());

    // Build the ModelRuntimeManager (will be moved into core-0)
    let mut maybe_manager = Some(ModelRuntimeManager::new(
        load_rx,
        starter_txs,
        loaded_models,
        metrics_registry.clone(),
    ));

    let mut handles = Vec::new();

    for core_id in 0..num_cores {
        let server = grpc_server.clone();
        let starter_rx = starter_rxs[core_id].take().unwrap();
        let metrics_for_worker = metrics_registry.clone();
        // Move manager into core-0's thread only
        let manager_for_core = if core_id == 0 {
            maybe_manager.take()
        } else {
            None
        };
        let metrics_for_core = if core_id == 0 {
            Some(metrics_registry.clone())
        } else {
            None
        };

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
                    // Initialize thread-local metrics and spawn periodic flush
                    metrics::init_local_metrics(metrics_for_worker);
                    compio::runtime::spawn(async {
                        loop {
                            compio::time::sleep(std::time::Duration::from_secs(1)).await;
                            metrics::flush_local_metrics();
                        }
                    })
                    .detach();

                    // Spawn the session starter for this core
                    compio::runtime::spawn(SessionStarter::new(starter_rx).run()).detach();

                    // Core-0 also runs the model runtime manager and metrics server
                    if let Some(m) = manager_for_core {
                        compio::runtime::spawn(m.run()).detach();
                    }
                    if let Some(mr) = metrics_for_core {
                        compio::runtime::spawn(metrics::serve_metrics("0.0.0.0:9090", mr)).detach();
                    }

                    serve(server).await
                })
                .unwrap();
            })
            .unwrap();
        handles.push(handle);

        // Break out of the loop after moving manager so the compiler knows it's consumed once
        if core_id == 0 {
            continue;
        }
    }

    // Send initial model load request
    let (reply_tx, _reply_rx) = tokio::sync::oneshot::channel();
    load_tx.blocking_send(LoadModelRequest {
        model_name: String::from("Int64ToFloat64Model"),
        version: 1,
        model_path: PathBuf::from("samples/model.onnx"),
        config: ModelRepositoryConfig {
            backend: Backend::Supertensor,
            batch_size: 256,
            capacity: 32,
            num_executors: 10,
            allocator: AllocatorKind::CudaPinned,
        },
        reply: reply_tx,
    })?;

    for h in handles {
        h.join().expect("worker thread panicked");
    }

    Ok(())
}
