from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor
from .recurrent_inputs import append_recurrent_state_inputs


class Qwen3NextProcessor(BasicLLMProcessor):
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

        append_recurrent_state_inputs(model_inputs, scheduler_output)
        return model_inputs
