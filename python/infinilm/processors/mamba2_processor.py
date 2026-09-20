import infinicore
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .mamba_processor import MambaProcessor
from .processor import register_processor


@register_processor("mamba2")
class Mamba2Processor(MambaProcessor):
    """Processor for Mamba2 models with one recurrent state row per request."""

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
            scheduler_output, temperature, top_p, top_k, **kwargs
        )
        init_indices = []
        final_indices = []
        for request in scheduler_output.scheduled_requests:
            state_index = request.mamba_cache_index
            if isinstance(scheduler_output, StaticSchedulerOutput):
                state_index = 1
            if state_index is None:
                raise RuntimeError(
                    f"Request {request.request_id} has no assigned Mamba2 state row"
                )
            init_indices.append(0 if scheduler_output.is_prefill else state_index)
            final_indices.append(state_index)

        model_inputs["mamba_init_state_indices"] = infinicore.from_list(
            init_indices, dtype=infinicore.int32
        )
        model_inputs["mamba_final_state_indices"] = infinicore.from_list(
            final_indices, dtype=infinicore.int32
        )
        model_inputs["block_tables"] = None
        model_inputs["slot_mapping"] = None
        return model_inputs
