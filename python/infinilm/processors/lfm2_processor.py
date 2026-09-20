"""Input/cache routing for LFM2's recurrent ShortConv layers."""

import infinicore
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


def static_short_conv_state_indices(is_prefill, prefix_hit_len, num_requests):
    """Reserve row 0 for zero history and row 1 for one active request."""
    if num_requests != 1:
        raise ValueError("LFM2 static scheduling currently supports one request")
    if is_prefill and prefix_hit_len:
        raise ValueError(
            "LFM2 prefix reuse requires a matching ShortConv state snapshot"
        )
    return ([0] if is_prefill else [1]), [1]


@register_processor("lfm2")
class Lfm2Processor(BasicLLMProcessor):
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

        # Static KV positions alone cannot reset recurrent convolution state.
        # Route new requests from zero history and decode from the active row.
        if isinstance(scheduler_output, StaticSchedulerOutput):
            init_indices, final_indices = static_short_conv_state_indices(
                scheduler_output.is_prefill,
                scheduler_output.prefix_hit_len,
                len(scheduler_output.scheduled_requests),
            )
            model_inputs["mamba_init_state_indices"] = infinicore.from_list(
                init_indices, dtype=infinicore.int32
            )
            model_inputs["mamba_final_state_indices"] = infinicore.from_list(
                final_indices, dtype=infinicore.int32
            )
            return model_inputs

        init_indices = []
        final_indices = []
        for request in scheduler_output.scheduled_requests:
            if request.mamba_cache_index is None:
                raise RuntimeError(
                    f"Request {request.request_id} has no ShortConv cache index"
                )
            init_indices.append(
                0 if scheduler_output.is_prefill else request.mamba_cache_index
            )
            final_indices.append(request.mamba_cache_index)

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs
