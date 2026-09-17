import gc
import logging
import os
import time

from infinilm.base_config import BaseConfig
from infinilm.llm.llm import LLM
from infinilm.moe_config import configure_moe_ep_backend
from infinilm.processors.videonsa_processor import decode_video_frames

DEFAULT_VIDEO_NUM_FRAMES = 8


def test(
    prompts: list[str],
    model_path,
    draft_model_path=None,
    num_draft_tokens=4,
    max_new_tokens=100,
    device="cpu",
    tp=1,
    pp=1,
    pp_stage=0,
    master_addr="127.0.0.1",
    master_port=29500,
    moe_ep_backend="disabled",
    ep=1,
    enable_paged_attn=False,
    enable_graph=False,
    num_blocks=512,
    block_size=256,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    attn_backend="default",
    use_mla=False,
    image_path=None,
    video_path=None,
    video_num_frames=None,
    skip_load=False,
    weight_load_mode="async",
    use_legacy_moe=False,
    enable_prefix_caching=True,
    pre_transpose=False,
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if top_k != 1:
        raise ValueError("Hugging Face token comparison requires --top-k=1")
    if image_path is not None or video_path is not None:
        raise ValueError("The Granite Hugging Face comparison supports text prompts only")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Running Hugging Face (static cache, greedy decoding)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)
    hf_model.eval()
    hf_results = []
    with torch.inference_mode():
        for prompt in prompts:
            hf_inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_token_ids = hf_inputs["input_ids"][0].tolist()
            hf_inputs = {name: tensor.to(device) for name, tensor in hf_inputs.items()}
            hf_sequences = hf_model.generate(
                **hf_inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                cache_implementation="static",
                disable_compile=True,
            )
            input_length = hf_inputs["input_ids"].shape[1]
            hf_token_ids = hf_sequences[0, input_length:].cpu().tolist()
            hf_results.append((prompt_token_ids, hf_token_ids))
            del hf_sequences, hf_inputs
    del hf_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = LLM(
        model_path=model_path,
        draft_model_path=draft_model_path,
        num_draft_tokens=num_draft_tokens,
        device=device,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        pipeline_parallel_stage=pp_stage,
        master_addr=master_addr,
        master_port=master_port,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=ep,
        cache_type="paged" if enable_paged_attn else "static",
        max_batch_size=len(prompts),
        max_tokens=max_new_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enable_graph=enable_graph,
        attn_backend=attn_backend,
        use_mla=use_mla,
        skip_load=skip_load,
        weight_load_mode=weight_load_mode,
        use_legacy_moe=use_legacy_moe,
        enable_prefix_caching=enable_prefix_caching,
        pre_transpose=pre_transpose,
    )

    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompts
    ]
    if video_path is not None:
        video_payload = decode_video_frames(
            video_path, video_num_frames or DEFAULT_VIDEO_NUM_FRAMES
        )
        for conversation in conversations:
            conversation[0]["content"] = [
                {"type": "video_url", "video_url": {"url": video_payload}}
            ] + conversation[0]["content"]
    elif image_path is not None:
        for conversation in conversations:
            conversation[0]["content"] = [
                {"type": "image_url", "image_url": {"url": image_path}}
            ] + conversation[0]["content"]

    t1 = time.time()
    print("=================== start generate ====================")

    try:
        outputs = model.chat(
            messages=conversations,
        )
    finally:
        model.close()
    t2 = time.time()

    all_matches = len(outputs) == len(hf_results)
    for i, output in enumerate(outputs):
        print(f"Resquest {i}:")
        print("===Query===")
        print(output.prompt)
        print("===Response===")
        print(output.outputs[0].text)
        print("")

        prompt_token_ids, hf_token_ids = hf_results[i]
        local_token_ids = output.outputs[0].token_ids
        prompt_matches = output.prompt_token_ids == prompt_token_ids
        mismatch = next(
            (
                index
                for index, (local_token, hf_token) in enumerate(
                    zip(local_token_ids, hf_token_ids)
                )
                if local_token != hf_token
            ),
            None,
        )
        if mismatch is None and len(local_token_ids) != len(hf_token_ids):
            mismatch = min(len(local_token_ids), len(hf_token_ids))
        exact_match = prompt_matches and mismatch is None
        all_matches = all_matches and exact_match

        print("=== Hugging Face ===")
        print(f"token_ids: {hf_token_ids}")
        print(tokenizer.decode(hf_token_ids, skip_special_tokens=True))
        print("=== InfiniLM ===")
        print(f"token_ids: {local_token_ids}")
        print("=== Comparison ===")
        print(f"prompt token IDs match: {prompt_matches}")
        print(f"generated token IDs match: {mismatch is None}")
        if mismatch is not None:
            hf_token = hf_token_ids[mismatch] if mismatch < len(hf_token_ids) else None
            local_token = (
                local_token_ids[mismatch] if mismatch < len(local_token_ids) else None
            )
            print(
                f"first mismatch at generated token {mismatch}: "
                f"Hugging Face={hf_token}, InfiniLM={local_token}"
            )
        print(f"RESULT: {'PASS' if exact_match else 'FAIL'}")
        print("")

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )

    if not all_matches:
        raise SystemExit(1)


if __name__ == "__main__":
    cfg = BaseConfig()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if cfg.pp > 1 and cfg.node_rank > 0:
        from infinilm.server.pipeline_worker import run_worker

        run_worker(cfg)
        raise SystemExit(0)

    device_str = cfg.get_device_str(cfg.device)

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    model_path = cfg.model

    max_new_tokens = cfg.max_new_tokens

    tp = cfg.tp

    enable_paged_attn = cfg.enable_paged_attn

    enable_graph = cfg.enable_graph

    if cfg.use_legacy_moe:
        moe_ep_backend, ep = "disabled", 1
    else:
        moe_ep_backend, ep = configure_moe_ep_backend(
            cfg.tp, cfg.dp, cfg.ep, cfg.moe_ep_backend, cfg.model
        )

    test(
        prompts,
        model_path,
        draft_model_path=cfg.draft_model,
        num_draft_tokens=cfg.num_draft_tokens,
        max_new_tokens=max_new_tokens,
        device=device_str,
        tp=tp,
        pp=cfg.pp,
        pp_stage=cfg.node_rank,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        moe_ep_backend=moe_ep_backend,
        ep=ep,
        enable_paged_attn=enable_paged_attn,
        enable_graph=enable_graph,
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        attn_backend=cfg.attn,
        use_mla=cfg.use_mla,
        image_path=cfg.image,
        video_path=cfg.video,
        video_num_frames=cfg.video_num_frames,
        skip_load=cfg.skip_load,
        weight_load_mode=cfg.weight_load_mode,
        use_legacy_moe=cfg.use_legacy_moe,
        enable_prefix_caching=cfg.enable_prefix_caching,
        pre_transpose=cfg.pre_transpose,
    )
