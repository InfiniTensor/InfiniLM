use llama_cu::SessionId;
use openai_struct::{
    ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
    CreateChatCompletionStreamResponseChoices, FinishReason,
};

const CHAT_COMPLETION_OBJECT: &str = "chat.completion.chunk";
pub(crate) const V1_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

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
