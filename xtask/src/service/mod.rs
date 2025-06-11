mod cache_manager;
mod error;
mod openai;
mod response;

use crate::{BaseArgs, service::openai::create_chat_completion_response};
use cache_manager::CacheManager;
use error::*;
use http_body_util::{BodyExt, combinators::BoxBody};
use hyper::{
    Method, Request, Response,
    body::{Bytes, Incoming},
    server::conn::http1,
    service::Service as HyperService,
};
use hyper_util::rt::TokioIo;
use llama_cu::{
    Message, Received, ReturnReason, SampleArgs, Service, SessionId, Terminal, TextBuf, utok,
};
use log::{debug, info, warn};
use openai::V1_CHAT_COMPLETIONS;
use openai_struct::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
    CreateChatCompletionRequest, FinishReason,
};
use response::{error, text_stream};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    ffi::c_int,
    future::Future,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    net::TcpListener,
    sync::mpsc::{self, UnboundedSender},
};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Args)]
pub struct ServiceArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    port: u16,
    #[clap(long)]
    think: bool,
}

impl ServiceArgs {
    pub fn service(self) {
        let Self { base, port, think } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(start_infer_service(
                base.model,
                port,
                gpus,
                max_steps,
                !base.no_cuda_graph,
                think,
            ))
            .unwrap()
    }
}

async fn start_infer_service(
    model: PathBuf,
    port: u16,
    gpus: Box<[c_int]>,
    max_steps: usize,
    use_cuda_graph: bool,
    think: bool,
) -> std::io::Result<()> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));
    info!("start service at {addr}");

    let mut service = Service::new(model, &gpus, use_cuda_graph);
    let sessions: BTreeMap<SessionId, SessionInfo> = BTreeMap::new();

    let (think, _think) = if think {
        let &[think] = &*service.terminal().encode("<think>") else {
            unreachable!()
        };
        let &[_think] = &*service.terminal().encode("</think>") else {
            unreachable!()
        };
        (think, _think)
    } else {
        (utok::MAX, utok::MAX)
    };

    let service_manager = Arc::new(ServiceManager {
        terminal: service.terminal().clone(),
        max_steps,
        sessions: Mutex::new(sessions),
        cache_manager: Mutex::new(CacheManager::new(service.terminal().clone())),
    });

    let service_manager_for_recv = service_manager.clone();

    let _response = tokio::task::spawn_blocking(move || {
        loop {
            let Received { sessions, outputs } = service.recv(Duration::from_millis(10));

            // 先处理输出
            for (session_id, tokens) in outputs {
                if tokens.is_empty() {
                    continue;
                }

                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let session_info = sessions_guard.get_mut(&session_id).unwrap();
                // 更新 session_info
                session_info.tokens.extend(&tokens);

                let mut tokens = &tokens[..];
                if tokens.first().is_some_and(|t| t == &think) {
                    session_info.think = true;
                    tokens = &tokens[1..]
                }
                let think = if session_info.think {
                    &tokens[..tokens.iter().take_while(|t| **t != _think).count()]
                } else {
                    &[]
                };

                if session_info.think {
                    if think.len() < tokens.len() {
                        session_info.think = false;
                        tokens = &tokens[think.len() + 1..]
                    } else {
                        tokens = &[]
                    }
                }
                let think = service_manager_for_recv
                    .terminal
                    .decode(think, &mut session_info.buf);
                let text = service_manager_for_recv
                    .terminal
                    .decode(tokens, &mut session_info.buf);
                debug!("解码完成：{tokens:?} -> {think:?} | {text:?}");

                let response = create_chat_completion_response(
                    session_id,
                    session_info.created as _,
                    session_info.model.clone(),
                    Some(think).filter(|s| !s.is_empty()),
                    Some(text).filter(|s| !s.is_empty()),
                    None,
                );
                let message = serde_json::to_string(&response).unwrap();

                if session_info.sender.send(message).is_err() {
                    info!("{session_id:?} 客户端连接已关闭");
                    service_manager_for_recv.terminal.stop(session_id);
                }
            }

            // 处理会话结束
            if !sessions.is_empty() {
                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let mut cache_manager_guard =
                    service_manager_for_recv.cache_manager.lock().unwrap();

                for (session, reason) in sessions {
                    let SessionInfo {
                        tokens,
                        sender,
                        model,
                        created,
                        ..
                    } = sessions_guard.remove(&session.id).unwrap();
                    let reason = match reason {
                        // 正常完成，插回cache
                        ReturnReason::Finish => {
                            cache_manager_guard.insert(tokens, session.cache);
                            info!("{:?} 正常完成", session.id);
                            FinishReason::Stop
                        }
                        ReturnReason::Overflow => {
                            info!("{:?} 超长完成", session.id);
                            FinishReason::Length
                        }
                    };
                    let response = create_chat_completion_response(
                        session.id,
                        created as i32,
                        model,
                        None,
                        None,
                        Some(reason),
                    );
                    sender
                        .send(serde_json::to_string(&response).unwrap())
                        .unwrap_or_else(|_| info!("{:?} 发送正常完成失败", session.id));
                }
            }
        }
    });

    let app = App(service_manager);

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

struct SessionInfo {
    sender: UnboundedSender<String>,
    buf: TextBuf,
    think: bool,
    tokens: Vec<utok>,
    model: String,
    created: u64,
}

struct ServiceManager {
    terminal: Terminal,
    max_steps: usize,
    sessions: Mutex<BTreeMap<SessionId, SessionInfo>>,
    cache_manager: Mutex<CacheManager>,
}

#[derive(Clone)]
struct App(Arc<ServiceManager>);

impl HyperService<Request<Incoming>> for App {
    type Response = Response<BoxBody<Bytes, hyper::Error>>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let service_manager = self.0.clone();
        match (req.method(), req.uri().path()) {
            (&Method::POST, V1_CHAT_COMPLETIONS) => Box::pin(async move {
                let whole_body = req.collect().await?.to_bytes();
                let req = serde_json::from_slice(&whole_body);
                Ok(match req {
                    Ok(completions) => complete_chat(completions, service_manager),
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

fn complete_chat(
    completions: CreateChatCompletionRequest,
    service_manager: Arc<ServiceManager>,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    let CreateChatCompletionRequest {
        model,
        messages,
        max_tokens,
        temperature,
        top_p,
        ..
    } = completions;
    let (sender, receiver) = mpsc::unbounded_channel();

    let max_steps = max_tokens.map_or(service_manager.max_steps, |n| n as usize);
    let sample_args =
        SampleArgs::new(temperature.unwrap_or(0.), top_p.unwrap_or(1.), usize::MAX).unwrap();

    debug!("received completions: {messages:#?}");

    // 用于持有所有权
    let mut content_list = Vec::with_capacity(messages.len());
    for msg in &messages {
        let msg = match msg {
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: Value::String(msg),
                ..
            }) => msg,
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: Value::String(msg),
                ..
            }) => msg,
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: Some(Value::String(msg)),
                ..
            }) => msg,
            msg => return error(Error::msg_not_supported(msg)),
        };
        content_list.push(msg)
    }

    let messages = messages
        .iter()
        .zip(&content_list)
        .map(|(message, content)| match message {
            ChatCompletionRequestMessage::User(_) => Message::user(content.as_str()),
            ChatCompletionRequestMessage::System(_) => Message::system(content.as_str()),
            ChatCompletionRequestMessage::Assistant(_) => Message::assistant(content.as_str()),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    debug!("received messages: {messages:#?}");
    let text = service_manager.terminal.render(&messages);
    debug!("received prompt: {text}");
    let tokens = service_manager.terminal.tokenize(&text);

    let (id, tokens) =
        service_manager
            .cache_manager
            .lock()
            .unwrap()
            .send(tokens, sample_args, max_steps);

    let session_info = SessionInfo {
        sender,
        tokens,
        buf: TextBuf::new(),
        think: false,
        model,
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    assert!(
        service_manager
            .sessions
            .lock()
            .unwrap()
            .insert(id, session_info,)
            .is_none()
    );

    text_stream(UnboundedReceiverStream::new(receiver))
}

#[cfg(test)]
mod client;
