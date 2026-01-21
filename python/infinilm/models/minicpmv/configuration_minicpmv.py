# coding=utf-8

import infinicore

from infinilm.lib import _infinilm
from ...configuration_utils import PretrainedConfig
from ..llama.configuration_llama import LlamaConfig


class MiniCPMVConfig(PretrainedConfig, _infinilm.MiniCPMVConfig):
    model_type = "minicpmv"

    def __init__(
        self,
        vision_config=None,
        query_num=64,
        drop_vision_last_layer=False,
        batch_vision_input=True,
        torch_dtype=None,
        **kwargs,
    ):
        _infinilm.MiniCPMVConfig.__init__(self)

        self.model_type = "minicpmv"

        if torch_dtype == "bfloat16":
            torch_dtype = "float16"

        llm_kwargs = dict(kwargs)
        for key in (
            "model_type",
            "vision_config",
            "query_num",
            "slice_config",
            "slice_mode",
            "batch_vision_input",
            "drop_vision_last_layer",
            "image_size",
            "patch_size",
            "version",
            "auto_map",
            "architectures",
        ):
            llm_kwargs.pop(key, None)
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
        if torch_dtype is not None:
            llm_kwargs.setdefault("torch_dtype", torch_dtype)
        self.llm_config = LlamaConfig(**llm_kwargs)

        if vision_config is None:
            vision_config = {}
        self.vision_config = _infinilm.SiglipVisionConfig()
        for key, value in vision_config.items():
            if value is None or not hasattr(self.vision_config, key):
                continue
            setattr(self.vision_config, key, value)

        self.query_num = query_num
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input

        if torch_dtype is None and "torch_dtype" in kwargs:
            torch_dtype = kwargs["torch_dtype"]

        if torch_dtype is not None:
            if torch_dtype in {"float32", "bfloat16", "float16"}:
                self.dtype = getattr(infinicore, torch_dtype)
                self._dtype = self.dtype._underlying
            else:
                raise ValueError(f"Unsupported dtype: {torch_dtype}")

        pad_token_id = kwargs.pop("pad_token_id", getattr(self.llm_config, "pad_token_id", None))
        bos_token_id = kwargs.pop("bos_token_id", getattr(self.llm_config, "bos_token_id", None))
        eos_token_id = kwargs.pop("eos_token_id", getattr(self.llm_config, "eos_token_id", None))
        tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", getattr(self.llm_config, "tie_word_embeddings", False)
        )

        self.pad_token_id = pad_token_id
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
