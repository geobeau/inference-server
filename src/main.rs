mod cli;
mod grpc;
mod loader;
mod metrics;
mod model_repository;
mod model_runtime;
mod scheduler;
mod tensor;
mod tracing;
use ::tracing::info;
use arc_swap::ArcSwap;
use clap::Parser;
use pajamax::{serve, Server};
use std::sync::Arc;

use hashbrown::HashMap;
use tracing_subscriber::EnvFilter;

use ort::{environment::GlobalThreadPoolOptions, execution_providers::{ArbitrarilyConfigurableExecutionProvider, CUDAExecutionProvider, ExecutionProviderDispatch, TensorRTExecutionProvider}};

use crate::{
    grpc::{inference::GrpcInferenceServiceServer, TritonService},
    model_repository::{LoadedModel, LocalModelRepository, ModelRepository},
    model_runtime::{LoadModelRequest, ModelRuntimeManager, SessionStarter},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_file(true)
        .with_line_number(true)
        .with_target(true)
        .init();

    let args = cli::Args::parse();
    let processing_cores = args.processing_cores;
    let executor_cores = args.executor_cores;

    // Validate CLI early so we fail fast before starting gRPC / workers
    let model_source = args.model_source();

    // Discover models before starting workers so errors surface immediately
    let discovered: Vec<LoadedModel> = match model_source {
        cli::ModelSource::Local(path) => {
            let repo = LocalModelRepository::new(path);
            repo.load_all()
                .expect("failed to load models from local directory")
        }
        cli::ModelSource::S3 {
            endpoint,
            bucket,
            prefix,
            region,
            cache_dir,
        } => {
            let repo = ModelRepository::new(&endpoint, &bucket, &prefix, &region, cache_dir);
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(repo.load_all())
                .expect("failed to load models from S3")
        }
    };
    let discovered: Vec<LoadedModel> = if let Some(ref filter) = args.load_models {
        let allowed: std::collections::HashSet<&str> = filter.iter().map(|s| s.as_str()).collect();
        discovered
            .into_iter()
            .filter(|m| allowed.contains(m.name.as_str()))
            .collect()
    } else {
        discovered
    };

    let providers: Vec<ExecutionProviderDispatch> = args
        .execution_providers
        .iter()
        .map(|ep| match ep {
            cli::ExecutionProviderKind::Cpu => {
                info!("Registering CPU execution provider");
                ort::execution_providers::CPUExecutionProvider::default()
                    .build()
            }
            cli::ExecutionProviderKind::Cuda => {
                info!("Registering CUDA execution provider");
                CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build()
                    .error_on_failure()
            }
            cli::ExecutionProviderKind::TensorRT => {
                info!("Registering TensorRT execution provider");
                let mut ep = TensorRTExecutionProvider::default()
                    .with_device_id(0);
                if let Some(min_subgraph_size) = args.trt_min_subgraph_size {
                    ep = ep.with_arbitrary_config(
                        "trt_min_subgraph_size",
                        min_subgraph_size.to_string(),
                    );
                }
                if let Some(ref exclude) = args.trt_op_types_to_exclude {
                    ep = ep.with_arbitrary_config("trt_op_types_to_exclude", exclude.join(","));
                }
                ep.build()
                    .error_on_failure()
            }
        })
        .collect();
    let thread_pool = GlobalThreadPoolOptions::default()
        .with_intra_threads(args.ort_intra_threads)
        .unwrap()
        .with_inter_threads(args.ort_inter_threads)
        .unwrap()
        .with_spin_control(false)
        .unwrap();
    ort::init()
        .with_execution_providers(providers)
        .with_global_thread_pool(thread_pool)
        .commit();

    // Metrics registry
    let metrics_registry = Arc::new(metrics::MetricsRegistry::new());

    // Shared model map for gRPC handlers
    let loaded_models: Arc<ArcSwap<HashMap<String, Arc<scheduler::ModelProxy>>>> =
        Arc::new(ArcSwap::from_pointee(HashMap::new()));

    // Create the load-model channel
    let (load_tx, load_rx) = tokio::sync::mpsc::channel::<LoadModelRequest>(16);

    let addr = &args.grpc_addr;
    info!("Starting Triton gRPC server on {}", addr);

    let config = pajamax::Config::new()
        .max_concurrent_connections(args.max_concurrent_connections)
        .max_concurrent_streams(args.max_concurrent_streams)
        .max_frame_size(args.max_frame_size);

    let grpc_loaded_models = loaded_models.clone();
    let services: Vec<Box<dyn Fn() -> std::rc::Rc<dyn pajamax::PajamaxService> + Send + Sync>> =
        vec![Box::new(move || {
            std::rc::Rc::new(GrpcInferenceServiceServer::new(TritonService::new(
                grpc_loaded_models.clone(),
            )))
        })];
    let grpc_server = Server::new(services, config, addr.to_string());

    // Create per-executor-core session starter channels
    let mut starter_txs = Vec::with_capacity(executor_cores);
    let mut starter_rxs = Vec::with_capacity(executor_cores);
    for _ in 0..executor_cores {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        starter_txs.push(tx);
        starter_rxs.push(Some(rx));
    }

    // Spawn the ModelRuntimeManager on its own dedicated thread
    let custom_op_libraries = args.custom_op_libraries.unwrap_or_default();
    let manager_metrics = metrics_registry.clone();
    let manager = ModelRuntimeManager::new(
        load_rx,
        starter_txs,
        loaded_models,
        manager_metrics.clone(),
        custom_op_libraries,
    );
    let manager_handle = std::thread::Builder::new()
        .name("model-manager".into())
        .spawn(move || {
            let rt = compio::runtime::RuntimeBuilder::new()
                .build()
                .expect("failed to build manager compio runtime");
            rt.block_on(async move {
                metrics::init_local_metrics(manager_metrics);
                compio::runtime::spawn(async {
                    loop {
                        compio::time::sleep(std::time::Duration::from_secs(1)).await;
                        metrics::flush_local_metrics();
                    }
                })
                .detach();
                manager.run().await;
            });
        })
        .unwrap();

    let mut handles = vec![manager_handle];

    // Spawn dedicated executor core threads
    for core_id in 0..executor_cores {
        let starter_rx = starter_rxs[core_id].take().unwrap();
        let metrics_for_executor = metrics_registry.clone();

        let handle = std::thread::Builder::new()
            .name(format!("exec-{core_id}"))
            .spawn(move || {
                let rt = compio::runtime::RuntimeBuilder::new()
                    .build()
                    .expect("failed to build executor compio runtime");
                rt.block_on(async move {
                    metrics::init_local_metrics(metrics_for_executor);
                    compio::runtime::spawn(async {
                        loop {
                            compio::time::sleep(std::time::Duration::from_secs(1)).await;
                            metrics::flush_local_metrics();
                        }
                    })
                    .detach();
                    SessionStarter::new(starter_rx).run().await;
                });
            })
            .unwrap();
        handles.push(handle);
    }

    // Spawn gRPC processing core threads
    for core_id in 0..processing_cores {
        let server = grpc_server.clone();
        let metrics_for_worker = metrics_registry.clone();
        let metrics_for_core = if core_id == 0 {
            Some(metrics_registry.clone())
        } else {
            None
        };

        let handle = std::thread::Builder::new()
            .name(format!("proc-{core_id}"))
            .spawn(move || {
                let mut proactor = compio::driver::ProactorBuilder::new();
                proactor
                    .capacity(8096)
                    .coop_taskrun(true)
                    .taskrun_flag(true);

                let rt = compio::runtime::RuntimeBuilder::new()
                    .with_proactor(proactor.to_owned())
                    .event_interval(1024)
                    .build()
                    .expect("failed to build compio runtime");

                rt.block_on(async move {
                    metrics::init_local_metrics(metrics_for_worker);
                    compio::runtime::spawn(async {
                        loop {
                            compio::time::sleep(std::time::Duration::from_secs(1)).await;
                            metrics::flush_local_metrics();
                        }
                    })
                    .detach();

                    if let Some(mr) = metrics_for_core {
                        compio::runtime::spawn(metrics::serve_metrics("0.0.0.0:9090", mr)).detach();
                    }

                    serve(server).await
                })
                .unwrap();
            })
            .unwrap();
        handles.push(handle);
    }

    // Dispatch discovered models to the runtime manager
    info!("Loading {} models", discovered.len());
    let mut load_replies = Vec::new();
    for model in discovered {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        load_tx.blocking_send(LoadModelRequest {
            model_name: model.name.clone(),
            version: model.version,
            model_path: model.model_path,
            config: model.config,
            reply: reply_tx,
        })?;
        load_replies.push((model.name, reply_rx));
    }
    for (name, reply_rx) in load_replies {
        match reply_rx.blocking_recv() {
            Ok(Ok(())) => info!("Model {} loaded successfully", name),
            Ok(Err(e)) => panic!("Failed to load model {}: {:?}", name, e),
            Err(_) => panic!("Model loader dropped without responding for {}", name),
        }
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }

    Ok(())
}
