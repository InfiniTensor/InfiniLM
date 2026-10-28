"""Native four-layer logits with independent images and graph decode.

Run in separate processes with optimizations disabled/enabled, then compare
saved arrays. Transformers is used only for image/token preprocessing.
"""

import argparse, ctypes
from pathlib import Path
import numpy as np
import torch, infinicore as ic
from PIL import Image, ImageDraw
from infinilm.infer_engine import InferEngine
from infinilm.cache import PagedKVCacheConfig
from infinilm.processors.qwen3_5_processor import Qwen35Processor
from infinilm.llm.scheduler import Scheduler
from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
import infinilm.modeling_utils as weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ic.set_device(ic.device("cuda", 0))
    check = weights.check_parameters
    weights.check_parameters = lambda keys, loaded: check(
        keys, list(set(keys) & set(loaded))
    )
    engine = InferEngine(
        args.model,
        device=ic.device("cuda", 0),
        cache_config=PagedKVCacheConfig(
            num_blocks=128, block_size=64, max_batch_size=8
        ),
        enable_graph_compiling=True,
        attention_backend="flash-attn",
        weight_load_mode="sync",
        dtype="bfloat16",
    )
    weights.load_model_state_dict_by_file(engine, args.model, engine.dtype)
    processor = Qwen35Processor(args.model)
    scheduler = Scheduler(
        max_batch_size=8,
        num_blocks=128,
        block_size=64,
        max_num_batched_tokens=8192,
        has_mamba_cache=True,
        num_mamba_cache_blocks=9,
        enable_prefix_caching=False,
    )
    requests = []
    for i, (width, height) in enumerate([(640, 480), (512, 384), (640, 480)]):
        im = Image.new("RGB", (width, height), (240 - i * 40, 210, 180 + i * 30))
        ImageDraw.Draw(im).text((20, 20), str(137 + i * 211), fill="black")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "test"}},
                    {"type": "text", "text": "Read the number in this picture."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        processed = processor(prompt, images=[im], return_tensors="pt")
        req = InferenceRequest(
            str(i),
            prompt=prompt,
            prompt_token_ids=processed["input_ids"].flatten().tolist(),
            processed_inputs=processed,
            has_multimodal_inputs=True,
            sampling_params=SamplingParams(
                max_tokens=8, temperature=0, top_k=1, top_p=1, ignore_eos=True
            ),
        )
        requests.append(req)
        scheduler.add_request(req)
    arrays = {}
    for step in range(5):
        batch = scheduler.schedule()
        data = processor.build_model_inputs(batch, temperature=0, top_k=1, top_p=1)

        # Keep the Python tensor wrappers alive: from_torch retains its
        # external backing storage there. _build_input unwraps for the call.
        data["sample_all_positions"] = False
        output = super(InferEngine, engine).forward(engine._build_input(**data))
        logits = ic.Tensor(output.logits).to(ic.device("cpu"))
        raw = ctypes.string_at(logits.data_ptr(), logits.numel() * 2)
        arrays[str(step) + "_logits"] = (
            torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
            .view(torch.bfloat16)
            .float()
            .numpy()
            .reshape(tuple(logits.shape))
        )
        ids = ic.Tensor(output.output_ids).to_numpy().ravel().tolist()
        arrays[str(step) + "_ids"] = np.asarray(ids)
        assert len(ids) == len(batch.scheduled_requests)
        for req, token in zip(batch.scheduled_requests, ids):
            scheduler.commit_computed_tokens(req, req.get_total_length())
            req.append_generated_token_id(int(token))
        scheduler.complete_requests(batch.scheduled_requests)
        assert np.isfinite(arrays[str(step) + "_logits"]).all()
        print("PASS_MULTIMODAL_FORWARD", step, ids, flush=True)
    for req in requests:
        req.status = RequestStatus.CANCELED
    scheduler.complete_requests(requests)
    assert scheduler.get_cache_stats()["num_used_mamba_cache_blocks"] == 0
    assert scheduler.get_cache_stats()["usable_blocks"] == 128
    np.savez_compressed(args.output, **arrays)
    print("MULTIMODAL_NATIVE_NUMERICS_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
