#![cfg(detected_neuware)]

mod resource;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{upos, utok, FileLoadError};
use common_cn::rustTensor as Tensor;
use std::path::Path;

pub use common_cn::{cndrv, synchronize};
pub use resource::Cache;

pub struct Transformer;

impl Model for Transformer {
    type Meta = ();
    type Error = FileLoadError;

    fn load(_model_dir: impl AsRef<Path>, _meta: Self::Meta) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    fn max_seq_len(&self) -> upos {
        todo!()
    }

    fn eos_token(&self) -> utok {
        todo!()
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        todo!()
    }

    fn duplicate_cache(&self, _cache: &Tensor<Self::Storage>, _pos: upos) -> Tensor<Self::Storage> {
        todo!()
    }

    fn token_embed(&self, _queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        todo!()
    }

    fn forward<'a>(
        &self,
        _queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        _token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        todo!()
    }

    fn decode(
        &self,
        _decoding: impl IntoIterator<Item = DecodingMeta>,
        _hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        todo!()
    }

    fn sample(
        &self,
        _args: impl IntoIterator<Item = SampleMeta>,
        _logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        todo!()
    }
}
