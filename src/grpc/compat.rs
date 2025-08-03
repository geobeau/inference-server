use ort::tensor::TensorElementType;

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
}
