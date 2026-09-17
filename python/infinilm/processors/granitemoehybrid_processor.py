import infinicore
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("granitemoehybrid")
class GraniteMoeHybridProcessor(BasicLLMProcessor):
    """Attach causal-convolution state rows to each Granite forward pass."""

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
        for request in scheduler_output.scheduled_requests:
            if isinstance(scheduler_output, StaticSchedulerOutput):
                state_index = 1
            else:
                state_index = request.mamba_cache_index
                if state_index is None:
                    raise RuntimeError(
                        f"Request {request.request_id} has no assigned mamba "
                        "cache index"
                    )

            init_indices.append(0 if scheduler_output.is_prefill else state_index)
            final_indices.append(state_index)

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        return model_inputs
