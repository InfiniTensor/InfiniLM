use super::{
    cache_manager::CacheManager,
    error::Error,
    response::{error, text_stream},
};
use crate::service::{ModelConfig, openai::create_chat_completion_response};
use http_body_util::combinators::BoxBody;
use hyper::{Response, body::Bytes};
use llama_cu::{
    Message, Received, ReturnReason, SampleArgs, Service, SessionId, Terminal, TextBuf, utok,
};
use log::{debug, info};
use openai_struct::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
    CreateChatCompletionRequest, FinishReason,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::mpsc::{self, UnboundedSender},
    task::JoinHandle,
};
use tokio_stream::wrappers::UnboundedReceiverStream;

pub(super) struct Model {
    terminal: Terminal,
    max_steps: usize,
    sessions: Mutex<BTreeMap<SessionId, SessionInfo>>,
    cache_manager: Mutex<CacheManager>,
}

struct SessionInfo {
    sender: UnboundedSender<String>,
    buf: TextBuf,
    think: bool,
    tokens: Vec<utok>,
    model: String,
    created: u64,
}

impl Model {
    pub fn new(config: ModelConfig, use_cuda_graph: bool) -> (Arc<Self>, JoinHandle<()>) {
        let ModelConfig {
            path,
            gpus,
            max_steps,
            think,
        } = config;

        let mut service = Service::new(path, &gpus.unwrap_or(Box::new([0])), use_cuda_graph);
        let sessions: BTreeMap<SessionId, SessionInfo> = BTreeMap::new();

        let (think, _think) = if think.unwrap_or(false) {
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

        let service_manager = Arc::new(Model {
            terminal: service.terminal().clone(),
            max_steps: max_steps.unwrap_or(2 << 10),
            sessions: Mutex::new(sessions),
            cache_manager: Mutex::new(CacheManager::new(service.terminal().clone())),
        });

        let service_manager_for_recv = service_manager.clone();

        let join_handle = tokio::task::spawn_blocking(move || {
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
                        if let Some(_think) = tokens.iter().position(|t| *t == _think) {
                            session_info.think = false;
                            let think = &tokens[.._think];
                            tokens = &tokens[_think + 1..];
                            think
                        } else {
                            let think = tokens;
                            tokens = &[];
                            think
                        }
                    } else {
                        &[]
                    };

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

        (service_manager, join_handle)
    }

    pub fn complete_chat(
        &self,
        completions: CreateChatCompletionRequest,
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

        let max_steps = max_tokens.map_or(self.max_steps, |n| n as usize);
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
                ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessage {
                        content: Some(Value::String(msg)),
                        ..
                    },
                ) => msg,
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
        let text = self.terminal.render(&messages);
        debug!("received prompt: {text}");
        let tokens = self.terminal.tokenize(&text);

        let (id, tokens) = self
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
            self.sessions
                .lock()
                .unwrap()
                .insert(id, session_info,)
                .is_none()
        );

        text_stream(UnboundedReceiverStream::new(receiver))
    }
}
