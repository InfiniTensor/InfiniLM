mod cache_manager;
mod error;
mod model;
mod openai;
mod response;

use crate::parse_gpus;
use error::*;
use http_body_util::{BodyExt, combinators::BoxBody};
use hyper::{
    Request, Response,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use log::{info, warn};
use model::Model;
use openai::create_models;
use openai_struct::CreateChatCompletionRequest;
use response::error;
use response::json;
use std::collections::HashMap;
use std::{ffi::c_int, fs::read_to_string, path::Path};
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    pin::Pin,
    sync::Arc,
};
use tokio::net::TcpListener;

#[derive(Args)]
pub struct ServiceArgs {
    file: String,

    #[clap(short, long)]
    port: u16,
    #[clap(long)]
    no_cuda_graph: bool,

    #[clap(long)]
    name: Option<String>,
    #[clap(long)]
    gpus: Option<String>,
    #[clap(long)]
    max_tokens: Option<usize>,
    #[clap(long)]
    think: bool,
}

#[derive(serde::Deserialize, Debug)]
pub struct ModelConfig {
    pub path: String,
    pub gpus: Option<Box<[c_int]>>,
    pub max_tokens: Option<usize>,
    pub think: Option<bool>,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self {
            file,
            port,
            no_cuda_graph,
            name,
            gpus,
            max_tokens,
            think,
        } = self;

        let path = Path::new(&file);
        let model_configs = match path.extension().map(|s| s.to_str()) {
            Some(Some("toml")) => toml::from_str(&read_to_string(path).unwrap()).unwrap(),
            Some(Some("gguf")) => [(
                name.as_deref()
                    .unwrap_or_else(|| path.file_stem().unwrap().to_str().unwrap())
                    .to_string(),
                ModelConfig {
                    path: file.clone(),
                    gpus: Some(parse_gpus(gpus.as_deref())),
                    max_tokens,
                    think: Some(think),
                },
            )]
            .into(),
            _ => panic!("file must be a gguf model or a toml config"),
        };

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(start_infer_service(model_configs, port, !no_cuda_graph))
            .unwrap()
    }
}

async fn start_infer_service(
    model_configs: HashMap<String, ModelConfig>,
    port: u16,
    use_cuda_graph: bool,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");
    info!("model_name list: {:?}", model_configs.keys());

    let mut handles = Vec::with_capacity(model_configs.len());
    let models = model_configs
        .into_iter()
        .map(|(name, config)| {
            let (model, handle) = Model::new(config, use_cuda_graph);
            handles.push(handle);
            (name, model)
        })
        .collect();

    let app = App(Arc::new(models));

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
        info!("ready to accept");
        let (stream, x) = listener.accept().await?;
        info!("listen from {x}");
        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), app)
                .await
            {
                warn!("Error serving connection: {err:?}")
            }
        });
    }
}

#[derive(Clone)]
struct App(Arc<HashMap<String, Arc<Model>>>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        match (req.method(), req.uri().path()) {
            openai::GET_MODELS => {
                let json = json(create_models(self.0.keys().cloned()));
                Box::pin(async move { Ok(json) })
            }
            openai::POST_CHAT_COMPLETIONS => {
                let models = self.0.clone();
                Box::pin(async move {
                    let whole_body = req.collect().await?.to_bytes();
                    let req = serde_json::from_slice::<CreateChatCompletionRequest>(&whole_body);
                    Ok(match req {
                        Ok(req) => match models.get(&req.model) {
                            Some(model) => model.complete_chat(req),
                            None => error(Error::ModelNotFound(req.model)),
                        },
                        Err(e) => error(Error::WrongJson(e)),
                    })
                })
            }
            // Return 404 Not Found for other routes.
            (method, uri) => {
                let msg = Error::not_found(method, uri);
                Box::pin(async move { Ok(error(msg)) })
            }
        }
    }
}

#[cfg(test)]
mod client;
