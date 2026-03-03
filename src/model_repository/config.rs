use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Supertensor,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocatorKind {
    Cpu,
    CudaPinned,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRepositoryConfig {
    pub backend: Backend,
    pub batch_size: usize,
    pub capacity: usize,
    pub num_executors: usize,
    pub allocator: AllocatorKind,
}
