"""Build recurrent-state indices shared by hybrid model processors."""


def append_recurrent_state_inputs(model_inputs, scheduler_output):
    from .. import _core as infinicore

    indices = []
    for request in scheduler_output.scheduled_requests:
        if request.mamba_cache_index is None:
            raise RuntimeError(
                f"Request {request.request_id} has no assigned mamba cache index."
            )
        indices.append(request.mamba_cache_index)
    model_inputs["mamba_init_state_indices"] = infinicore.from_list(
        [0] * len(indices) if scheduler_output.is_prefill else indices,
        dtype=infinicore.int32,
    )
    model_inputs["mamba_final_state_indices"] = infinicore.from_list(
        indices,
        dtype=infinicore.int32,
    )
