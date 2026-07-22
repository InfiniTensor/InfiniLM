import logging

from infinilm.base_config import BaseConfig
from infinilm.config.engine_config import EngineConfig
from infinilm.distributed.pipeline_transport import PipelineWorkerClient
from infinilm.llm.model_runner.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def run_worker(cfg: BaseConfig) -> None:
    if cfg.pp <= 1 or cfg.node_rank == 0:
        raise ValueError(
            "pipeline worker mode requires --pp > 1 and --node-rank in [1, pp)"
        )

    config = EngineConfig(
        model_path=cfg.model,
        device=cfg.get_device_str(cfg.device),
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        pipeline_parallel_size=cfg.pp,
        pipeline_parallel_stage=cfg.node_rank,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        cache_type="paged" if cfg.enable_paged_attn else "static",
        max_batch_size=cfg.max_batch_size,
        max_tokens=cfg.max_new_tokens,
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        max_cache_len=cfg.max_cache_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        enable_graph=cfg.enable_graph,
        attn_backend=cfg.attn,
        use_mla=cfg.use_mla,
        weight_load_mode=cfg.weight_load_mode,
        skip_load=cfg.skip_load,
        skip_legacy_moe=cfg.skip_legacy_moe,
    )

    runner = ModelRunner(config, initialize_processor=False)
    worker = PipelineWorkerClient(
        runner,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        pp_stage=cfg.node_rank,
    )
    logger.info(
        "Pipeline worker event loop started: role=worker, stage=%s, coordinator=%s:%s",
        cfg.node_rank,
        cfg.master_addr,
        cfg.master_port,
    )
    try:
        worker.serve_forever()
    finally:
        runner.close()


def main() -> None:
    cfg = BaseConfig()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_worker(cfg)


if __name__ == "__main__":
    main()
