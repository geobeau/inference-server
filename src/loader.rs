use std::{clone, collections::HashMap};

use futures::{future::Map, stream::StreamExt};

use ndarray::{stack, Axis};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{Input, Output, Session, SessionInputValue, SessionOutputs},
    tensor::{Shape, TensorElementType},
    value::{
        DynTensor, DynTensorValueType, DynValue, Tensor, TensorArrayData, Value, ValueRef,
        ValueType,
    },
};

use crate::{
    grpc::inference::batch_output,
    scheduler::{InferenceRequest, InferenceResponse},
};

pub struct OnnxExecutor {
    pub session: Session,
    pub inputs: flume::Receiver<InferenceRequest>, // <SessionInputs<'a, 'a>>,
}

pub struct BatchableTensor {
    pub inner_tensor: DynTensor,
    pub offset: usize,
    pub data_type: TensorElementType,
    pub shape: Shape,
}

pub struct BatchedTensor {
    pub inner_tensor: DynValue,
    pub offset: usize,
    pub data_type: TensorElementType,
    tensor_shape: Shape,
}

macro_rules! copy_tensor_slice {
    ($ty:ty, $inner:expr, $tensor_to_append:expr, $offset:expr) => {{
        let (new_shape, new_data) = $tensor_to_append.try_extract_tensor::<$ty>().unwrap();
        let new_offset = $offset + new_shape.num_elements();
        let (_, data) = $inner.try_extract_tensor_mut::<$ty>().unwrap();
        data[$offset..new_offset].copy_from_slice(new_data);
        new_offset
    }};
}

macro_rules! pop_tensor_slice {
    ($ty:ty, $inner:expr, $new_tensor:expr, $offset:expr, $new_offset:expr) => {{
        let (_, data) = $new_tensor.try_extract_tensor_mut::<$ty>().unwrap();
        let (_, inner_data) = $inner.try_extract_tensor::<$ty>().unwrap();
        data.copy_from_slice(&inner_data[$new_offset..$offset]);
    }};
}

pub struct BatchableInputs {
    pub inputs: HashMap<String, BatchableTensor>,
}

impl BatchableInputs {
    fn new(inputs: &Vec<Input>, batch_size: i64) -> BatchableInputs {
        let mut inner_inputs = HashMap::with_capacity(inputs.len());
        inputs.iter().for_each(|input| {
            let shape = input.input_type.tensor_shape().unwrap();
            let data_type = input.input_type.tensor_type().unwrap();
            inner_inputs.insert(input.name.clone(), BatchableTensor::new(data_type, shape, batch_size));
        });

        BatchableInputs {
            inputs: inner_inputs,
        }
    }

    fn append_inputs(&mut self, inputs: HashMap<String, DynTensor>) {
        inputs.iter().for_each(|(name, tensor)| {
            self.inputs.get_mut(name).unwrap().append_raw_tensor(tensor);
        });
    }

    fn session_inputs_view(&self) -> HashMap<String, ValueRef<'_, DynTensorValueType>> {
        let mut all_inputs = HashMap::new();

        self.inputs.iter().for_each(|(name, batch_tensor)| {
            all_inputs.insert(name.clone(), batch_tensor.inner_tensor.view());
        });
        return all_inputs;
    }

    fn clear(&mut self) {
        self.inputs.iter_mut().for_each(|(_, input)| {
            input.clear();
        });
    }
}

pub struct BatchedOutputs {
    pub outputs: HashMap<String, BatchedTensor>,
}

impl BatchedOutputs {
    fn new(outputs: SessionOutputs) -> BatchedOutputs {
        let mut inner_ouputs = HashMap::with_capacity(outputs.len());
        outputs.into_iter().for_each(|(name, dyn_value)| {
            let batched_tensor = BatchedTensor::from(dyn_value);
            inner_ouputs.insert(String::from(name), batched_tensor);
        });

        BatchedOutputs {
            outputs: inner_ouputs,
        }
    }

    fn pop_outputs(&mut self) -> HashMap<String, DynTensor> {
        let mut outputs = HashMap::new();

        self.outputs.iter_mut().for_each(|(name, batch_tensor)| {
            outputs.insert(name.clone(), batch_tensor.pop());
        });
        return outputs;
    }
}

impl BatchedTensor {
    fn from(dyn_value: DynValue) -> BatchedTensor {
        let mut shape = dyn_value.shape().clone();
        let offset = shape.num_elements();
        println!("ouput: {shape}");
        // the shape of each sub tensor
        shape[0] = 1;
        BatchedTensor {
            data_type: dyn_value.data_type().clone(),
            tensor_shape: shape,
            inner_tensor: dyn_value,
            offset,
        }
    }

    fn pop(&mut self) -> DynTensor {
        println!("pop once");
        let new_offset = self.offset - self.tensor_shape.num_elements();
        let mut tensor = DynTensor::new(
            &Allocator::default(),
            self.data_type,
            self.tensor_shape.clone(),
        )
        .unwrap();
        match self.data_type {
            TensorElementType::Float32 => {
                pop_tensor_slice!(f32, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Uint8 => {
                pop_tensor_slice!(u8, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Int8 => {
                pop_tensor_slice!(i8, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Uint16 => {
                pop_tensor_slice!(u16, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Int16 => {
                pop_tensor_slice!(i16, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Int32 => {
                pop_tensor_slice!(i32, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Int64 => {
                pop_tensor_slice!(i64, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Bool => {
                pop_tensor_slice!(bool, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Float64 => {
                pop_tensor_slice!(f64, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Uint32 => {
                pop_tensor_slice!(u32, self.inner_tensor, tensor, self.offset, new_offset)
            }
            TensorElementType::Uint64 => {
                pop_tensor_slice!(u64, self.inner_tensor, tensor, self.offset, new_offset)
            }

            // Unsupported or special handling types
            TensorElementType::Float16 => todo!(),
            TensorElementType::Bfloat16 => todo!(),
            TensorElementType::Complex64 => todo!(),
            TensorElementType::Complex128 => todo!(),
            TensorElementType::Float8E4M3FN => todo!(),
            TensorElementType::Float8E4M3FNUZ => todo!(),
            TensorElementType::Float8E5M2 => todo!(),
            TensorElementType::Float8E5M2FNUZ => todo!(),
            TensorElementType::Uint4 => todo!(),
            TensorElementType::Int4 => todo!(),
            TensorElementType::String => todo!(),
            TensorElementType::Undefined => todo!(),
        };
        self.offset = new_offset;
        return tensor;
    }
}

impl BatchableTensor {
    fn new(data_type: TensorElementType, shape: &Shape, batch_size: i64) -> BatchableTensor {
        let mut batch_shape = shape.clone();
        assert_eq!(shape[0], -1); // Ensure the shape is dynamic
        batch_shape[0] = batch_size;
        BatchableTensor {
            inner_tensor: DynTensor::new(&Allocator::default(), data_type, batch_shape).unwrap(),
            data_type,
            shape: shape.clone(),
            offset: 0,
        }
    }

    fn clear(&mut self) {
        self.offset = 0;
    }

    fn append_raw_tensor(&mut self, tensor: &DynTensor) {
        self.offset = match self.data_type {
            TensorElementType::Float32 => {
                copy_tensor_slice!(f32, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Uint8 => {
                copy_tensor_slice!(u8, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Int8 => {
                copy_tensor_slice!(i8, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Uint16 => {
                copy_tensor_slice!(u16, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Int16 => {
                copy_tensor_slice!(i16, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Int32 => {
                copy_tensor_slice!(i32, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Int64 => {
                copy_tensor_slice!(i64, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Bool => {
                copy_tensor_slice!(bool, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Float64 => {
                copy_tensor_slice!(f64, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Uint32 => {
                copy_tensor_slice!(u32, self.inner_tensor, tensor, self.offset)
            }
            TensorElementType::Uint64 => {
                copy_tensor_slice!(u64, self.inner_tensor, tensor, self.offset)
            }

            // Unsupported or special handling types
            TensorElementType::Float16 => todo!(),
            TensorElementType::Bfloat16 => todo!(),
            TensorElementType::Complex64 => todo!(),
            TensorElementType::Complex128 => todo!(),
            TensorElementType::Float8E4M3FN => todo!(),
            TensorElementType::Float8E4M3FNUZ => todo!(),
            TensorElementType::Float8E5M2 => todo!(),
            TensorElementType::Float8E5M2FNUZ => todo!(),
            TensorElementType::Uint4 => todo!(),
            TensorElementType::Int4 => todo!(),
            TensorElementType::String => todo!(),
            TensorElementType::Undefined => todo!(),
        };
    }
}

impl OnnxExecutor {
    pub async fn run(&mut self) {
        println!("executor started");
        let mut stream = self.inputs.stream();
        let batch_size = 2;
        let mut batched_requests_chan = Vec::new();
        let mut batched_requests_tracing = Vec::new();

        let mut input_batch = BatchableInputs::new(&self.session.inputs, batch_size as i64);

        let (ty, mut shape) = match &self.session.outputs[0].output_type {
            ValueType::Tensor { ty, shape, .. } => (ty.clone(), shape.clone()),
            ValueType::Sequence(_) => todo!(),
            ValueType::Map { .. } => todo!(),
            ValueType::Optional(_) => todo!(),
        };
        shape[0] = batch_size as i64;
        println!("{:?}", shape);
        loop {
            // Batch requests
            while let Some(mut req) = stream.next().await {
                req.tracing.executor_start = Some(req.tracing.start.elapsed());
                input_batch.append_inputs(req.inputs);
                batched_requests_chan.push(req.resp_chan);
                batched_requests_tracing.push(req.tracing);
                if batched_requests_chan.len() == batch_size {
                    break;
                }
            }

            let session_outputs = self.session.run(input_batch.session_inputs_view()).unwrap();
            let mut batched_tensor = BatchedOutputs::new(session_outputs);

            while let Some(response_chan) = batched_requests_chan.pop() {
                let outputs = batched_tensor.pop_outputs();

                let trace = batched_requests_tracing.pop().unwrap();
                let response = InferenceResponse {
                    outputs: outputs,
                    tracing: trace,
                };
                response_chan.send_async(response).await.unwrap();
            }
            input_batch.clear();
        }
    }
}
