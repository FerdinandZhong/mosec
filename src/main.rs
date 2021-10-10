mod args;
mod coordinator;
mod errors;
mod metrics;
mod protocol;
mod tasks;

use std::net::SocketAddr;

use bytes::Bytes;
use clap::Clap;
use hyper::service::{make_service_fn, service_fn};
use hyper::{body::to_bytes, header::HeaderValue, Body, Method, Request, Response, StatusCode};
use prometheus::{Encoder, TextEncoder};
use tokio::signal::unix::{signal, SignalKind};
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::args::Opts;
use crate::coordinator::Coordinator;
use crate::errors::ServiceError;
use crate::metrics::Metrics;
use crate::tasks::{TaskCode, TaskManager};

const SERVER_INFO: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
const RESPONSE_NOT_FOUND: &str = "not found";
const RESPONSE_EMPTY: &str = "no data provided";
const RESPONSE_INTERNAL_ERROR: &str = "inference internal error";
const RESPONSE_UNKNOWN_ERROR: &str = "mosec unknown error";
const RESPONSE_SHUTDOWN: &str = "gracefully shutting down";

async fn index(_: Request<Body>) -> Response<Body> {
    let task_manager = TaskManager::global();
    if task_manager.is_shutdown() {
        return build_response(
            StatusCode::SERVICE_UNAVAILABLE,
            Bytes::from(RESPONSE_SHUTDOWN),
        );
    }
    build_response(StatusCode::OK, Bytes::from("MOSEC service"))
}

async fn metrics(_: Request<Body>) -> Response<Body> {
    let encoder = TextEncoder::new();
    let metrics = prometheus::gather();
    let mut buffer = vec![];
    encoder.encode(&metrics, &mut buffer).unwrap();
    build_response(StatusCode::OK, Bytes::from(buffer))
}

async fn inference(req: Request<Body>) -> Response<Body> {
    let task_manager = TaskManager::global();
    let data = to_bytes(req.into_body()).await.unwrap();
    let metrics = Metrics::global();

    if data.is_empty() {
        return build_response(StatusCode::OK, Bytes::from(RESPONSE_EMPTY));
    }

    metrics.remaining_task.inc();
    let mut status = StatusCode::OK;
    let mut content = Bytes::from(RESPONSE_UNKNOWN_ERROR);

    match task_manager.submit_task(data).await {
        Ok(task) => match task.code {
            TaskCode::Normal => {
                // Record latency only for successful tasks
                metrics
                    .duration
                    .with_label_values(&["total", "total"])
                    .observe(task.create_at.elapsed().as_secs_f64());
                content = task.data;
            }
            TaskCode::BadRequestError => {
                status = StatusCode::BAD_REQUEST;
                content = task.data; // Customized error message
            }
            TaskCode::ValidationError => {
                status = StatusCode::UNPROCESSABLE_ENTITY;
                content = task.data; // Customized error message
            }
            TaskCode::InternalError => {
                status = StatusCode::INTERNAL_SERVER_ERROR;
                content = Bytes::from(RESPONSE_INTERNAL_ERROR);
            }
            TaskCode::UnknownError => {
                status = StatusCode::NOT_IMPLEMENTED;
            }
        },
        Err(err) => {
            // Handle errors for which task cannot be retrieved
            content = Bytes::from(err.to_string());
            match err {
                ServiceError::TooManyRequests => {
                    status = StatusCode::TOO_MANY_REQUESTS;
                }
                ServiceError::Timeout => {
                    status = StatusCode::REQUEST_TIMEOUT;
                }
                ServiceError::UnknownError => {
                    status = StatusCode::NOT_IMPLEMENTED;
                }
                ServiceError::GracefulShutdown => {
                    status = StatusCode::SERVICE_UNAVAILABLE;
                }
            }
        }
    }
    metrics.remaining_task.dec();
    metrics
        .throughput
        .with_label_values(&[status.as_str()])
        .inc();

    build_response(status, content)
}

fn build_response(status: StatusCode, content: Bytes) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("server", HeaderValue::from_static(SERVER_INFO))
        .body(Body::from(content))
        .unwrap()
}

async fn service_func(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => Ok(index(req).await),
        (&Method::GET, "/metrics") => Ok(metrics(req).await),
        (&Method::POST, "/inference") => Ok(inference(req).await),
        _ => Ok(build_response(
            StatusCode::NOT_FOUND,
            Bytes::from(RESPONSE_NOT_FOUND),
        )),
    }
}

async fn shutdown_signal() {
    let mut interrupt = signal(SignalKind::interrupt()).unwrap();
    let mut terminate = signal(SignalKind::terminate()).unwrap();
    loop {
        tokio::select! {
            _ = interrupt.recv() => {
                info!("received interrupt signal and ignored at controller side");
            },
            _ = terminate.recv() => {
                info!("received terminate signal");
                let task_manager = TaskManager::global();
                task_manager.shutdown().await;
                info!("shutdown complete");
                break;
            },
        };
    }
}

fn init_env() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info")
    }

    tracing_subscriber::fmt::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

#[tokio::main]
async fn main() {
    init_env();
    let opts: Opts = Opts::parse();
    info!(?opts, "parse arguments");

    let coordinator = Coordinator::init_from_opts(&opts);
    tokio::spawn(async move {
        coordinator.run().await;
    });

    let service = make_service_fn(|_| async { Ok::<_, hyper::Error>(service_fn(service_func)) });
    let addr: SocketAddr = format!("{}:{}", opts.address, opts.port).parse().unwrap();
    let server = hyper::Server::bind(&addr).serve(service);
    let graceful = server.with_graceful_shutdown(shutdown_signal());
    if let Err(err) = graceful.await {
        tracing::error!(%err, "server error");
    }
}
