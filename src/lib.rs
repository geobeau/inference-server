// use std::{
//     io::{self, Write},
//     path::Path,
// };

// use ort::{
//     execution_providers::ExecutionProviderDispatch, inputs, session::{builder::GraphOptimizationLevel, Session}, value::TensorRef, Error
// };
// use rand::Rng;

// pub struct MatmulExecutor {
//     session: Session,
// }

// impl MatmulExecutor {
//     pub fn new(provider: ExecutionProviderDispatch) -> Result<MatmulExecutor, Error> {
//         let session = Session::builder()?
//             .with_optimization_level(GraphOptimizationLevel::Level1)?
// 			.with_execution_providers([provider])?
//             .with_intra_threads(1)?
//             .commit_from_file("samples/matmul.onnx")?;

// 		return Ok(MatmulExecutor{
// 			session,
// 		});
//     }

// 	pub fn run(&self, input_a: &ndarray::Array2::<f32>, input_b: &ndarray::Array2::<f32>) -> ort::Result<()> {
//         let input_a_value = Value::from_array(&self.session.allocator(), &input_a)?;
// let input_b_value = Value::from_array(&self.session.allocator(), &input_b)?;
// let inputs = ort::inputs![input_a_value, input_b_value];
// let _ = self.session.run(inputs)?;
// 		let _ = self.session.run(ort::inputs![input_a.view(), input_b.view()]?)?;

// 		Ok(())
// 	}
// }
