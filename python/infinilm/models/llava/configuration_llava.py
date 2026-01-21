# coding=utf-8

import infinicore

from infinilm.lib import _infinilm
from ...configuration_utils import PretrainedConfig
from ..llama.configuration_llama import LlamaConfig


class LlavaConfig(PretrainedConfig, _infinilm.LlavaConfig):
    model_type = "llava"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_index=32000,
        pad_token_id=0,
        vocab_size=32064,
        ignore_index=-100,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        tie_word_embeddings=False,
        torch_dtype=None,
        **kwargs,
    ):
        _infinilm.LlavaConfig.__init__(self)

        self.model_type = "llava"

        if text_config is None:
            text_config = {}
        if isinstance(text_config, LlamaConfig):
            self.text_config = text_config
        else:
            text_config = dict(text_config)
            text_config = {k: v for k, v in text_config.items() if v is not None}
            text_config.setdefault("attention_bias", False)
            text_config.setdefault("attention_output_bias", False)
            text_config.setdefault("mlp_bias", False)
            text_config.setdefault("pad_token_id", pad_token_id)
            if torch_dtype is not None:
                text_config.setdefault("torch_dtype", torch_dtype)
            self.text_config = LlamaConfig(**text_config)

        if vision_config is None:
            vision_config = {}
        self.vision_config = _infinilm.ClipVisionConfig()
        for key, value in vision_config.items():
            if value is None or not hasattr(self.vision_config, key):
                continue
            setattr(self.vision_config, key, value)

        self.image_token_index = image_token_index
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.tie_word_embeddings = tie_word_embeddings

        if torch_dtype is None and "torch_dtype" in kwargs:
            torch_dtype = kwargs["torch_dtype"]

        if torch_dtype is not None:
            if torch_dtype in {"float32", "bfloat16", "float16"}:
                self.dtype = getattr(infinicore, torch_dtype)
                self._dtype = self.dtype._underlying
            else:
                raise ValueError(f"Unsupported dtype: {torch_dtype}")

        bos_token_id = kwargs.pop("bos_token_id", getattr(self.text_config, "bos_token_id", None))
        eos_token_id = kwargs.pop("eos_token_id", getattr(self.text_config, "eos_token_id", None))
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", tie_word_embeddings)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        PretrainedConfig.__init__(
            self,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
