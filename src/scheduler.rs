use std::collections::HashMap;

use ort::value::Shape;

use crate::{
    grpc::inference::{model_metadata_response::TensorMetadata, ModelConfig},
    tensor::supertensor::SuperTensorBuffer,
};

#[derive(Clone)]
pub struct ModelMetadata {
    pub input_meta: Vec<TensorMetadata>,
    pub output_meta: Vec<TensorMetadata>,
    pub input_set: HashMap<String, Shape>,
}

pub struct ModelProxy {
    pub data: SuperTensorBuffer,
    pub model_config: ModelConfig,
    pub model_metadata: ModelMetadata,
}
