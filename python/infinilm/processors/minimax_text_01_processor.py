"""Processor for MiniMax-Text-01: assigns Lightning attention state slots.

MiniMax-Text-01 mixes full attention with Lightning (linear) attention layers.
The Lightning layers keep a recurrent state in a shared pool; this processor
tells the engine which pool slot each request reads from at the start of a
forward (`mamba_init_state_indices`) and writes to at the end
(`mamba_final_state_indices`), mirroring the Kimi-K3 / Qwen3Next processors.
"""

import infinicore
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("minimax_text_01")
class MiniMaxText01Processor(BasicLLMProcessor):
    @override
    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        model_inputs = super().build_model_inputs(
            scheduler_output,
            temperature,
            top_p,
            top_k,
            **kwargs,
        )

        init_indices = []
        final_indices = []
        for req in scheduler_output.scheduled_requests:
            if req.mamba_cache_index is None:
                raise RuntimeError(
                    f"Request {req.request_id} has no assigned mamba cache index"
                )
            init_indices.append(
                0 if scheduler_output.is_prefill else req.mamba_cache_index
            )
            final_indices.append(req.mamba_cache_index)

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs
