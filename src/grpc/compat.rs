use ndarray::IxDyn;
use ort::{
    tensor::TensorElementType,
    value::{DynTensor, Tensor},
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
    pub fn to_metadata_string(&self) -> String {
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

    pub fn from_str(s: &String) -> Self {
        match s.as_str() {
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

pub fn dyntensor_from_bytes(
    data_type: DataType,
    dimensions: &[usize],
    bytes: &Vec<u8>,
) -> DynTensor {
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
