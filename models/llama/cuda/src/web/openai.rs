use serde::{Deserialize, Serialize};

pub const V1_COMPLETIONS: &str = "/v1/completions";
pub const V1_COMPLETIONS_OBJECT: &str = "chat.completion";

/// <https://www.openaicto.com/api-reference/completions>
#[derive(Serialize, Deserialize)]
pub struct Completions {
    pub model: String,
    pub prompt: String,
}
#[derive(Serialize, Deserialize)]
pub struct CompletionsResponse {
    pub id: String,
    pub choices: Vec<CompletionsChoice>,
    pub created: usize,
    pub model: String,
    pub object: String,
}
#[derive(Serialize, Deserialize)]
pub struct CompletionsChoice {
    pub index: usize,
    pub text: String,
}
