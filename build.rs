fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(true)
        .out_dir("src/grpc")
        .compile_protos(
            &["proto/grpc_service.proto", "proto/model_config.proto"],
            &["proto"],
        )?;
    Ok(())
}
