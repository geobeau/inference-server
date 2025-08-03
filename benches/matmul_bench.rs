// use criterion::{criterion_group, criterion_main, Criterion};
// use ort::execution_providers::{CPUExecutionProvider, OpenVINOExecutionProvider};
// use ortest::MatmulExecutor;
// use rand::Rng;

// fn bench_matmul(c: &mut Criterion) {
//     let mut rng = rand::rng();
//     let cpu_provider = CPUExecutionProvider::default().build();
//     let vino_provider = OpenVINOExecutionProvider::default()
//         .with_device_type("GPU")
//         .build();

//     // CPU benchmark
//     let cpu_matmul = MatmulExecutor::new(cpu_provider).unwrap();

//     let input_a = ndarray::Array2::<f32>::from_shape_fn((1024, 1024), |_| rng.random::<f32>());
//     let input_b = ndarray::Array2::<f32>::from_shape_fn((1024, 1024), |_| rng.random::<f32>());
//     c.bench_function("matmul_cpu", |b| {
//         b.iter(|| {
//             cpu_matmul.run(&input_a, &input_b).unwrap();
//         })
//     });

//     // OpenVINO benchmark
//     let vino_matmul = MatmulExecutor::new(vino_provider).unwrap();

//     c.bench_function("matmul_openvino", |b| {
//         b.iter(|| {
//             vino_matmul.run(&input_a, &input_b).unwrap();
//         })
//     });
// }

// criterion_group!(benches, bench_matmul);
// criterion_main!(benches);
