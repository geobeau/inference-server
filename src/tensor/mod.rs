pub struct BatchableTensor {
    pub inner_tensor: DynTensor,
    pub offset: usize,
    pub data_type: TensorElementType,
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
    pub fn new(inputs: &[Input], batch_size: i64) -> BatchableInputs {
        let mut inner_inputs = HashMap::with_capacity(inputs.len());
        inputs.iter().for_each(|input| {
            let shape = input.input_type.tensor_shape().unwrap();
            let data_type = input.input_type.tensor_type().unwrap();
            inner_inputs.insert(
                input.name.clone(),
                BatchableTensor::new(data_type, shape, batch_size),
            );
        });

        BatchableInputs {
            inputs: inner_inputs,
        }
    }

    pub fn append_inputs(&mut self, inputs: HashMap<String, DynTensor>) {
        inputs.iter().for_each(|(name, tensor)| {
            self.inputs.get_mut(name).unwrap().append_raw_tensor(tensor);
        });
    }

    pub fn session_inputs_view(&self) -> HashMap<String, ValueRef<'_, DynTensorValueType>> {
        let mut all_inputs = HashMap::new();

        self.inputs.iter().for_each(|(name, batch_tensor)| {
            all_inputs.insert(name.clone(), batch_tensor.inner_tensor.view());
        });
        all_inputs
    }

    pub fn clear(&mut self) {
        self.inputs.iter_mut().for_each(|(_, input)| {
            input.clear();
        });
    }
}

pub struct BatchedOutputs {
    pub outputs: HashMap<String, BatchedTensor>,
}

impl BatchedOutputs {
    pub fn new(outputs: SessionOutputs) -> BatchedOutputs {
        let mut inner_ouputs = HashMap::with_capacity(outputs.len());
        outputs.into_iter().for_each(|(name, dyn_value)| {
            let batched_tensor = BatchedTensor::from(dyn_value);
            inner_ouputs.insert(String::from(name), batched_tensor);
        });

        BatchedOutputs {
            outputs: inner_ouputs,
        }
    }

    pub fn pop_outputs(&mut self) -> HashMap<String, DynTensor> {
        let mut outputs = HashMap::new();

        self.outputs.iter_mut().for_each(|(name, batch_tensor)| {
            outputs.insert(name.clone(), batch_tensor.pop());
        });
        outputs
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
            data_type: *dyn_value.data_type(),
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
        tensor
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

use std::collections::HashMap;

use ndarray::IxDyn;
use ort::{
    memory::Allocator,
    session::{Input, SessionOutputs},
    tensor::{Shape, TensorElementType},
    value::{DynTensor, DynTensorValueType, DynValue, Tensor, ValueRef},
};

use crate::grpc::inference::DataType;

impl From<TensorElementType> for DataType {
    fn from(val: TensorElementType) -> Self {
        match val {
            TensorElementType::Float32 => DataType::TypeFp32,
            TensorElementType::Uint8 => DataType::TypeUint8,
            TensorElementType::Int8 => DataType::TypeInt8,
            TensorElementType::Uint16 => DataType::TypeUint16,
            TensorElementType::Int16 => DataType::TypeInt16,
            TensorElementType::Int32 => DataType::TypeInt32,
            TensorElementType::Int64 => DataType::TypeInt64,
            TensorElementType::String => DataType::TypeString,
            TensorElementType::Bool => DataType::TypeBool,
            TensorElementType::Float16 => DataType::TypeFp16,
            TensorElementType::Float64 => DataType::TypeFp64,
            TensorElementType::Uint32 => DataType::TypeUint32,
            TensorElementType::Uint64 => DataType::TypeUint64,
            TensorElementType::Bfloat16 => DataType::TypeBf16,
            TensorElementType::Complex64 => DataType::TypeInvalid,
            TensorElementType::Complex128 => DataType::TypeInvalid,
            TensorElementType::Float8E4M3FN => DataType::TypeInvalid,
            TensorElementType::Float8E4M3FNUZ => DataType::TypeInvalid,
            TensorElementType::Float8E5M2 => DataType::TypeInvalid,
            TensorElementType::Float8E5M2FNUZ => DataType::TypeInvalid,
            TensorElementType::Uint4 => DataType::TypeInvalid,
            TensorElementType::Int4 => DataType::TypeInvalid,
            TensorElementType::Undefined => DataType::TypeInvalid,
        }
    }
}

impl DataType {
    pub fn to_metadata_string(self) -> String {
        match self {
            Self::TypeInvalid => String::from("INVALID"),
            Self::TypeBool => String::from("BOOL"),
            Self::TypeUint8 => String::from("UINT8"),
            Self::TypeUint16 => String::from("UINT16"),
            Self::TypeUint32 => String::from("UINT32"),
            Self::TypeUint64 => String::from("UINT64"),
            Self::TypeInt8 => String::from("INT8"),
            Self::TypeInt16 => String::from("INT16"),
            Self::TypeInt32 => String::from("INT32"),
            Self::TypeInt64 => String::from("INT64"),
            Self::TypeFp16 => String::from("FP16"),
            Self::TypeFp32 => String::from("FP32"),
            Self::TypeFp64 => String::from("FP64"),
            Self::TypeString => String::from("STRING"),
            Self::TypeBf16 => String::from("BF16"),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "INVALID" => Self::TypeInvalid,
            "BOOL" => Self::TypeBool,
            "INT8" => Self::TypeInt8,
            "UINT8" => Self::TypeUint8,
            "INT16" => Self::TypeInt16,
            "UINT16" => Self::TypeUint16,
            "INT32" => Self::TypeInt32,
            "UINT32" => Self::TypeUint32,
            "INT64" => Self::TypeInt64,
            "UINT64" => Self::TypeUint64,
            "FP16" => Self::TypeFp16,
            "BF16" => Self::TypeBf16,
            "FP32" => Self::TypeFp32,
            "FP64" => Self::TypeFp64,
            "STRING" => Self::TypeString,
            _ => Self::TypeInvalid,
        }
    }
}

pub fn dyntensor_from_bytes(data_type: DataType, dimensions: &[usize], bytes: &[u8]) -> DynTensor {
    match data_type {
        DataType::TypeInvalid => todo!(),
        DataType::TypeBool => {
            let data = bytes.iter().map(|d| (*d != 0)).collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeUint8 => {
            let data = bytes.to_vec();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeUint16 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 2]| u16::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeUint32 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 4]| u32::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeUint64 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 8]| u64::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeInt8 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 1]| i8::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeInt16 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 2]| i16::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeInt32 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 4]| i32::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeInt64 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 8]| {
                    let data = i64::from_le_bytes(*d);
                    println!("input {}", data);
                    data
                })
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeFp16 => todo!(),
        DataType::TypeFp32 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 4]| f32::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeFp64 => {
            let data = bytes
                .array_chunks()
                .map(|d: &[u8; 8]| f64::from_le_bytes(*d))
                .collect();
            let array = ndarray::Array::from_shape_vec(IxDyn(dimensions), data).unwrap();
            Tensor::from_array(array).unwrap().upcast()
        }
        DataType::TypeString => todo!(),
        DataType::TypeBf16 => todo!(),
    }
}
