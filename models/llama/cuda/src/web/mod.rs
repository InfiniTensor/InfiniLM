mod error;
mod infer;
mod openai;
mod response;

use error::Error;
use http_body_util::{combinators::BoxBody, BodyExt};
use hyper::{
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
    Method, Request, Response,
};
use hyper_util::rt::TokioIo;
use log::{info, warn};
use openai::{Completions, CompletionsChoice, CompletionsResponse, V1_COMPLETIONS_OBJECT};
use response::{error, single, text_stream};
use std::{
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::net::TcpListener;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[test]
fn service() {
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(start_infer_service())
        .unwrap()
}

enum Command {
    Infer {
        prompt: String,
        max_steps: usize,
        temperature: f32,
    },
    Stop,
}

enum InferResponse {
    Piece(String),
    Over,
}

async fn start_infer_service() -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 8000));
    println!("start service at {addr}");

    let (sender, commands) = mpsc::channel();
    let (responds, receiver) = mpsc::channel();
    let app = App {
        sender,
        receiver: Arc::new(Mutex::new(receiver)),
    };

    let _thread = std::thread::spawn(move || infer::infer(commands, responds));

    let listener = TcpListener::bind(addr).await?;
    loop {
        let app = app.clone();
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
struct App {
    sender: Sender<Command>,
    receiver: Arc<Mutex<Receiver<InferResponse>>>,
}

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let app = self.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, openai::V1_COMPLETIONS) => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body);
                Ok(match req {
                    Ok(completions) => complete(completions, app),
                    Err(e) => error(Error::WrongJson(e)),
                })
            }),
            // Return 404 Not Found for other routes.
            (method, uri) => {
                let msg = Error::not_found(method, uri);
                Box::pin(async move { Ok(error(msg)) })
            }
        }
    }
}

fn complete(completions: Completions, app: App) -> Response<BoxBody<Bytes, hyper::Error>> {
    let Completions {
        model,
        prompt,
        max_tokens,
        temperature,
        stream,
    } = completions;
    app.sender
        .send(Command::Infer {
            prompt,
            max_steps: max_tokens.unwrap_or(20000),
            temperature: temperature.unwrap_or(0.),
        })
        .unwrap();

    static ID: AtomicUsize = AtomicUsize::new(0);
    let id = format!("InfiniLM-{:#x}", ID.fetch_add(1, SeqCst));
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as _;

    if stream.is_some_and(|b| b) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        tokio::task::spawn_blocking(move || {
            let response = app.receiver.lock().unwrap();

            while let Ok(InferResponse::Piece(piece)) = response.recv() {
                let response = CompletionsResponse {
                    id: id.clone(),
                    choices: vec![CompletionsChoice {
                        index: 0,
                        text: piece,
                    }],
                    created,
                    model: model.clone(),
                    object: V1_COMPLETIONS_OBJECT.into(),
                };
                let msg = serde_json::to_string(&response).unwrap();
                if sender.send(msg).is_err() {
                    let _ = app.sender.send(Command::Stop);
                    break;
                }
            }
            app.sender.send(Command::Stop).unwrap()
        });
        text_stream(UnboundedReceiverStream::new(receiver))
    } else {
        let response = app.receiver.lock().unwrap();
        let mut text = String::new();
        while let Ok(InferResponse::Piece(piece)) = response.recv() {
            text.push_str(&piece)
        }
        let response = CompletionsResponse {
            id: id.clone(),
            choices: vec![CompletionsChoice { index: 0, text }],
            created,
            model: model.clone(),
            object: V1_COMPLETIONS_OBJECT.into(),
        };
        single(serde_json::to_string(&response).unwrap())
    }
}
