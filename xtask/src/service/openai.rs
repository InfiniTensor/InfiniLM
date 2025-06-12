use hyper::Method;
use llama_cu::SessionId;
use openai_struct::{
    ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
    CreateChatCompletionStreamResponseChoices, FinishReason, Model,
};
use serde::Serialize;

const CHAT_COMPLETION_OBJECT: &str = "chat.completion.chunk";
pub(crate) const GET_MODELS: (&Method, &str) = (&Method::GET, "models");
pub(crate) const POST_CHAT_COMPLETIONS: (&Method, &str) = (&Method::POST, "/chat/completions");

pub(crate) fn create_models(models: impl IntoIterator<Item = String>) -> impl Serialize {
    #[derive(Serialize)]
    struct Response {
        object: &'static str,
        data: Vec<Model>,
    }

    Response {
        object: "list",
        data: models
            .into_iter()
            .map(|id| Model {
                id,
                object: "model".into(),
                owned_by: "QYLab".into(),
                created: 0,
            })
            .collect(),
    }
}

pub(crate) fn create_chat_completion_response(
    id: SessionId,
    created: i32,
    model: String,
    think: Option<String>,
    answer: Option<String>,
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionStreamResponse {
    let choices = vec![CreateChatCompletionStreamResponseChoices {
        delta: ChatCompletionStreamResponseDelta {
            reasoning_content: think,
            content: answer,
            function_call: None,
            refusal: None,
            role: None,
            tool_calls: None,
        },
        finish_reason,
        index: 0,
        logprobs: None,
    }];
    CreateChatCompletionStreamResponse {
        id: format!("InfiniLM-Service-chatcmpl-{:#08x}", id.0),
        object: CHAT_COMPLETION_OBJECT.to_string(),
        created,
        model,
        choices,
        system_fingerprint: None,
        usage: None,
        service_tier: None,
    }
}
