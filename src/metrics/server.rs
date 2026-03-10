use std::sync::Arc;

use compio::buf::BufResult;
use compio::io::{AsyncRead, AsyncWriteExt};
use compio::net::{TcpListener, TcpStream};

use tracing::info;

use super::registry::MetricsRegistry;

async fn handle_connection(mut stream: TcpStream, registry: Arc<MetricsRegistry>) {
    let buf = Vec::with_capacity(1024);
    let BufResult(result, _buf) = stream.read(buf).await;
    if result.is_err() {
        return;
    }

    let body = registry.encode();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body,
    );

    let BufResult(_result, _) = stream.write_all(response.into_bytes()).await;
}

pub async fn serve_metrics(addr: &str, registry: Arc<MetricsRegistry>) {
    let listener = TcpListener::bind(addr).await.unwrap();
    info!(addr, "Metrics server listening");

    loop {
        let (stream, _peer_addr) = listener.accept().await.unwrap();
        let registry = registry.clone();
        compio::runtime::spawn(handle_connection(stream, registry)).detach();
    }
}
