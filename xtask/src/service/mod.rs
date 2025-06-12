mod cache_manager;
mod error;
mod openai;
mod response;

use crate::service::openai::create_chat_completion_response;
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
use std::collections::HashMap;

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
use tokio::{net::TcpListener, sync::mpsc::UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Args)]
pub struct ServiceArgs {
    #[clap(long)]
    model_path: Vec<PathBuf>,
    #[clap(long)]
    model_name: Vec<String>,
    #[clap(long)]
    gpus: Vec<String>,
    #[clap(long)]
    max_steps: Vec<usize>,
    #[clap(long)]
    think: Vec<bool>,
    #[clap(long)]
    no_cuda_graph: bool,
    #[clap(short, long)]
    port: u16,
}

pub type ModelConfig = (PathBuf, Box<[c_int]>, usize, bool);

impl ServiceArgs {
    pub fn service(self) {
        let Self {
            no_cuda_graph,
            port,
            ..
        } = self;

        let model_configs = self.get_model_configs();

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(start_infer_service(model_configs, port, !no_cuda_graph))
            .unwrap()
    }

    pub fn get_model_configs(&self) -> HashMap<String, ModelConfig> {
        let mut configs = HashMap::new();
        let model_name = if self.model_name.len() < self.model_path.len() {
            let mut names = self.model_name.clone();
            names.extend((names.len()..self.model_path.len()).map(|i| format!("model_{i}")));
            names
        } else {
            self.model_name[..self.model_path.len()].to_vec()
        };

        let gpus = if self.gpus.len() < self.model_path.len() {
            let mut gpus = self.gpus.clone();
            gpus.extend(std::iter::repeat_n(
                "0".to_string(),
                self.model_path.len() - gpus.len(),
            ));
            gpus
        } else {
            self.gpus[..self.model_path.len()].to_vec()
        };

        let max_steps = if self.max_steps.len() < self.model_path.len() {
            let mut steps = self.max_steps.clone();
            steps.extend(std::iter::repeat_n(
                512,
                self.model_path.len() - steps.len(),
            ));
            steps
        } else {
            self.max_steps[..self.model_path.len()].to_vec()
        };
        let think = if self.think.len() < self.model_path.len() {
            let mut think = self.think.clone();
            think.extend(std::iter::repeat_n(
                false,
                self.model_path.len() - think.len(),
            ));
            think
        } else {
            self.think[..self.model_path.len()].to_vec()
        };
        for ((((path, name), gpus), max_steps), think) in self
            .model_path
            .iter()
            .zip(model_name.iter())
            .zip(gpus.iter())
            .zip(max_steps.iter())
            .zip(think.iter())
        {
            let name = name.clone();
            let gpus: Box<[c_int]> = gpus.split(',').map(|s| s.parse().unwrap()).collect();
            configs.insert(name, (path.clone(), gpus, *max_steps, *think));
        }
        configs
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

    // Initialize terminals and cache managers for all models
    let mut terminals = BTreeMap::new();
    let mut max_steps_map = BTreeMap::new();
    let mut cache_managers = BTreeMap::new();
    let mut services = Vec::new();
    let mut think_map = BTreeMap::new();

    // Create a global channel for receiving messages from all services
    let (global_sender, global_receiver) = std::sync::mpsc::channel::<(String, Received)>();

    for (model_name, model_config) in model_configs {
        let model = model_config.0;
        let gpus = model_config.1;
        let max_steps = model_config.2;
        let think = model_config.3;
        let service = Service::new(model, &gpus, use_cuda_graph);

        let terminal = service.terminal().clone();
        services.push((model_name.clone(), service));
        terminals.insert(model_name.clone(), terminal.clone());
        max_steps_map.insert(model_name.clone(), max_steps);
        cache_managers.insert(model_name.clone(), CacheManager::new(terminal.clone()));
        let (think, _think) = if think {
            let &[think] = &*terminal.encode("<think>") else {
                unreachable!()
            };
            let &[_think] = &*terminal.encode("</think>") else {
                unreachable!()
            };
            (think, _think)
        } else {
            (utok::MAX, utok::MAX)
        };
        think_map.insert(model_name.clone(), (think, _think));
    }

    let sessions: BTreeMap<SessionId, SessionInfo> = BTreeMap::new();

    let service_manager = Arc::new(ServiceManager {
        terminal: terminals,
        max_steps_map,
        sessions: Mutex::new(sessions),
        cache_manager: Mutex::new(cache_managers),
    });

    // Spawn tasks to forward messages from each service to the global channel
    for (model_name, mut service) in services.into_iter() {
        let sender = global_sender.clone();

        std::thread::spawn(move || {
            loop {
                let received = service.recv(Duration::from_millis(50));
                debug!("Received from {model_name}");
                if sender.send((model_name.clone(), received)).is_err() {
                    debug!("Failed to send to global channel {model_name}");
                    break;
                }
            }
        });
    }

    let service_manager_for_recv = service_manager.clone();
    std::thread::spawn(move || {
        while let Ok((model_name, received)) = global_receiver.recv() {
            debug!("Received from global channel {model_name}");
            let Received { sessions, outputs } = received;

            // 先处理输出
            for (session_id, tokens) in outputs {
                if tokens.is_empty() {
                    continue;
                }

                let mut sessions_guard = service_manager_for_recv.sessions.lock().unwrap();
                let session_info = sessions_guard.get_mut(&session_id).unwrap();
                // 更新 session_info
                session_info.tokens.extend(&tokens);

                let terminal = service_manager_for_recv.terminal.get(&model_name).unwrap();
                let (think, _think) = think_map.get(&model_name).unwrap();
                let (think, text) = if *think != utok::MAX {
                    let mut tokens = &tokens[..];
                    if tokens.first().is_some_and(|t| t == think) {
                        session_info.think = true;
                        tokens = &tokens[1..]
                    }
                    let think = if session_info.think {
                        &tokens[..tokens.iter().take_while(|t| **t != *_think).count()]
                    } else {
                        &[]
                    };
                    if think.len() < tokens.len() {
                        session_info.think = false;
                        tokens = &tokens[think.len() + 1..]
                    } else {
                        tokens = &[]
                    }

                    let think = terminal.decode(think, &mut session_info.buf);
                    let text = terminal.decode(tokens, &mut session_info.buf);
                    debug!("解码完成：{tokens:?} -> {think:?} | {text:?}");
                    (think, text)
                } else {
                    let text = terminal.decode(&tokens, &mut session_info.buf);
                    debug!("解码完成：{tokens:?} -> {text:?}");
                    (String::from(""), text)
                };

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
                    terminal.stop(session_id);
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
                            cache_manager_guard
                                .get_mut(&model_name)
                                .unwrap()
                                .insert(tokens, session.cache);
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
    terminal: BTreeMap<String, Terminal>,
    max_steps_map: BTreeMap<String, usize>,
    sessions: Mutex<BTreeMap<SessionId, SessionInfo>>,
    cache_manager: Mutex<BTreeMap<String, CacheManager>>,
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
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();

    // Get the terminal for the requested model
    let terminal = match service_manager.terminal.get(&model) {
        Some(t) => t,
        None => return error(Error::ModelNotFound(model.clone())),
    };

    // Get the cache manager for the requested model
    let mut cache_manager_guard = service_manager.cache_manager.lock().unwrap();
    let cache_manager = match cache_manager_guard.get_mut(&model) {
        Some(cm) => cm,
        None => return error(Error::ModelNotFound(model)),
    };

    // Get max_steps for the requested model
    let max_steps = max_tokens.map_or_else(
        || {
            service_manager
                .max_steps_map
                .get(&model)
                .copied()
                .unwrap_or(2048)
        },
        |n| n as usize,
    );

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
    let text = terminal.render(&messages);
    debug!("received prompt: {text}");
    let tokens = terminal.tokenize(&text);

    let (id, tokens) = cache_manager.send(tokens, sample_args, max_steps);
    debug!("send to cache: {id:?}, {tokens:?}");

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
