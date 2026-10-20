import json
import os

MOE_EP_BACKENDS = {
    "disabled",
    "local_allreduce",
    "allgather_reducescatter",
}

MOE_EP_BACKEND_HELP = (
    "MoE expert-parallel backend: disabled keeps all experts on each rank; "
    "local_allreduce uses the tensor-parallel group as EP for TP=EP; "
    "allgather_reducescatter is reserved for true DP/EP routing; "
    "auto selects disabled for EP=1, local_allreduce for DP=1 and EP>1, "
    "and allgather_reducescatter for DP>1."
)


def normalize_moe_ep_backend(backend: str) -> str:
    backend = (backend or "auto").strip().lower()
    aliases = {
        "": "auto",
        "none": "disabled",
        "off": "disabled",
        "standard": "disabled",
        "0": "disabled",
        "naive": "allgather_reducescatter",
        "ag_rs": "allgather_reducescatter",
        "all_gather_reduce_scatter": "allgather_reducescatter",
        "local_all_reduce": "local_allreduce",
        "tp_ep": "local_allreduce",
        "vllm_tp": "local_allreduce",
        "dp1": "local_allreduce",
        "deep_ep": "deepep",
    }
    return aliases.get(backend, backend)


def is_moe_model(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path, "r") as f:
        config = json.load(f)
    model_type = str(config.get("model_type", "")).lower()
    return "moe" in model_type or "num_experts" in config


def configure_moe_ep_backend(
    tp: int,
    dp: int,
    ep: int | None,
    backend: str,
    model_path: str,
) -> tuple[str, int]:
    if tp < 1:
        raise ValueError("--tp must be greater than 0")
    if dp < 1:
        raise ValueError("--dp must be greater than 0")
    if not is_moe_model(model_path):
        return "disabled", 1

    if ep is None:
        ep = tp
    if ep < 1:
        raise ValueError("--ep must be greater than 0")

    backend = normalize_moe_ep_backend(backend)
    if backend == "auto":
        if dp == 1:
            backend = "disabled" if ep == 1 else "local_allreduce"
        else:
            backend = "allgather_reducescatter"
    if backend == "deepep":
        raise NotImplementedError(
            "DeepEP MoE EP backend is reserved but not enabled yet."
        )

    if backend not in MOE_EP_BACKENDS:
        raise ValueError(f"Unsupported --moe-ep-backend: {backend}")

    if dp != 1 and backend != "disabled":
        raise NotImplementedError(
            "InfiniLM currently has only a TP communication group. "
            "True DP>1 MoE EP needs DP rank/group support before selecting "
            f"{backend}."
        )
    if backend != "disabled" and ep != tp:
        raise NotImplementedError(
            "InfiniLM MoE EP currently reuses the TP communication group, "
            f"so EP size must equal TP size. Got EP={ep}, TP={tp}."
        )

    return backend, ep
