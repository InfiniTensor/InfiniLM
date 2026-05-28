# Copyright (c) 2025, InfiniCore
"""vLLM-compatible compile wrapper around ``TorchLlamaPrefillModel``."""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import VllmConfig

from infinilm.torch_llama import TorchLlamaPrefillModel, load_torch_llama


class VllmPrefillBackbone(nn.Module, TorchCompileWrapperWithCustomDispatcher):
    """
    Prefill backbone whose ``__call__`` goes through vLLM's torch.compile wrapper
    (single ``VllmBackend`` invocation per instance).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model_path: str,
        *,
        prefix: str = "prefill_backbone",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        cpp_state_dict: dict | None = None,
    ):
        nn.Module.__init__(self)
        TorchCompileWrapperWithCustomDispatcher.__init__(
            self, compilation_level=vllm_config.compilation_config.level
        )
        self.vllm_config = vllm_config
        self.prefix = prefix
        self._model: TorchLlamaPrefillModel = load_torch_llama(
            model_path,
            device=device,
            dtype=dtype,
            splitting_flash_boundary=True,
            cpp_state_dict=cpp_state_dict,
        )
        # PIECEWISE leaves ``use_custom_dispatcher`` false; enable bytecode dispatch
        # after the first ``compiled_callable`` pass populates ``compiled_codes``.
        self.use_custom_dispatcher = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.forward_prefill_compile(input_ids)

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # nn.Module.__call__ would bypass torch.compile; mirror vLLM decorator.
        if torch.compiler.is_compiling():
            return self.forward(input_ids, **kwargs)

        # First call: Dynamo + VllmBackend once. Later calls: direct bytecode dispatch.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            torch._dynamo.mark_dynamic(input_ids, 1)
            torch._dynamo.eval_frame.remove_from_cache(self.original_code_object)
            return self.compiled_callable(input_ids, **kwargs)

        with self.dispatch_to_code(0):
            return self.forward(input_ids, **kwargs)

    @property
    def inner(self) -> TorchLlamaPrefillModel:
        return self._model
