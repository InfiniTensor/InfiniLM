import infinicore

from .mamba_processor import MambaProcessor
from .processor import register_processor


@register_processor("mamba2")
class Mamba2Processor(MambaProcessor):
    def build_model_inputs(self, scheduler_output, *args, **kwargs):
        inputs = super().build_model_inputs(scheduler_output, *args, **kwargs)
        initial, final = [], []
        for request in scheduler_output.scheduled_requests:
            index = request.mamba_cache_index
            if index is None or index <= 0:
                raise RuntimeError("Mamba-2 requires an allocated nonzero state row.")
            initial.append(0 if request.num_local_cached_tokens == 0 else index)
            final.append(index)
        if len(set(final)) != len(final):
            raise RuntimeError("Mamba-2 requests cannot share a writable state row.")
        inputs["mamba_init_state_indices"] = infinicore.from_list(
            initial, dtype=infinicore.int32
        )
        inputs["mamba_final_state_indices"] = infinicore.from_list(
            final, dtype=infinicore.int32
        )
        return inputs
