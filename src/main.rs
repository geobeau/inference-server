mod grpc;
mod loader;
mod scheduler;

use std::collections::HashMap;

use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynTensor, Tensor},
};
use rand::Rng;
use tonic::transport::Server;

use crate::grpc::{inference::grpc_inference_service_server::GrpcInferenceServiceServer, TritonService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:8001".parse()?; // Triton default gRPC port is 8001
                                        // Initialize our service with S3 connection details
    let service = TritonService::new();
    println!("Starting Triton gRPC server on {}", addr);
    Server::builder()
        .add_service(GrpcInferenceServiceServer::new(service))
        .serve(addr)
        .await?;

    let (input_tx, input_rx) = flume::bounded(4096);

    let mut executor_endpoints = Vec::new();
    for _ in 0..4 {
        let (tx, rx) = flume::bounded(128);
        let cpu_provider = CPUExecutionProvider::default().build();
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_execution_providers([cpu_provider])
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_file("samples/matmul.onnx")
            .unwrap();
        let mut executor = loader::OnnxExecutor {
            session,
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
    };
    tokio::spawn(async move {
        sched.run().await;
    });

    for _ in 0..4 {
        let mut rng = rand::rng();
        let input_a = ndarray::Array2::<f32>::from_shape_fn((1024, 1024), |_| rng.random::<f32>());
        let input_b = ndarray::Array2::<f32>::from_shape_fn((1024, 1024), |_| rng.random::<f32>());

        let data1 = Tensor::from_array(input_a).unwrap().upcast();
        let data2 = Tensor::from_array(input_b).unwrap().upcast();

        let mut data: HashMap<String, DynTensor> = HashMap::with_capacity(2);

        data.insert(String::from("A"), data1);
        data.insert(String::from("B"), data2);

        match input_tx.send(data) {
            Ok(_) => println!("success"),
            Err(x) => println!("Error {}", x),
        }
    }
    Ok(())
}
