#!/usr/bin/env python3
"""End-to-end Gemma 4 parity & verification suite for InfiniLM.

Verifies token-for-token greedy parity against HuggingFace Gemma4ForConditionalGeneration:
  1. Text generation (short chat + math) on real gemma-4-E2B-it
  2. Long-context (>512 tokens) sliding-window + full-attention retrieval
  3. Vision multimodal inference (16-layer ViT + 2D Axial RoPE + 3x3 Pooler + ClippableLinear)
  4. Audio multimodal inference (12-layer Conformer + Conv2d subsampling + Causal LightConv1d)
  5. Gemma 4 26B-A4B MoE architecture (parallel dense MLP + 3D GeGLU experts + k_eq_v) for TP=1 and TP=2
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
import torch
from PIL import Image, ImageDraw


def verify_e2b(model_dir: str, device: str, max_new_tokens: int = 24) -> None:
    import soundfile as sf
    from infinilm.llm.llm import LLM
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_dir)
    hf_device = "cuda:0" if device == "cuda" else "cpu"
    hf_model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=hf_device
    ).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "red_rect.png")
        img = Image.new("RGB", (240, 192), color=(255, 255, 255))
        ImageDraw.Draw(img).rectangle([40, 30, 200, 160], fill=(220, 20, 20))
        img.save(img_path)

        wav_path = os.path.join(tmpdir, "tone.wav")
        sr = 16000
        t = np.linspace(0, 1.5, int(1.5 * sr), dtype=np.float32)
        wav = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(wav_path, wav, sr)

        long_prompt = (
            "Remember the secret code: ALPHA-7392.\n"
            + ("The quick brown fox jumps over the lazy dog. " * 65)
            + "\nWhat was the secret code? Reply with just the code."
        )

        cases = [
            ("text_chat", [{"role": "user", "content": "How are you?"}], None, None),
            (
                "text_math",
                [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
                None,
                None,
            ),
            (
                "long_sliding_686",
                [{"role": "user", "content": long_prompt}],
                None,
                None,
            ),
            (
                "image_red_rect",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": "What color is the rectangle in this image? Reply in one short sentence.",
                            },
                        ],
                    }
                ],
                [img],
                None,
            ),
            (
                "audio_tone",
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio"},
                            {
                                "type": "text",
                                "text": "Describe what you hear in this audio clip in one short sentence.",
                            },
                        ],
                    }
                ],
                None,
                [wav],
            ),
        ]

        hf_expected = {}
        with torch.no_grad():
            for name, conv, imgs, auds in cases:
                txt = proc.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                kwargs = {"text": txt, "return_tensors": "pt"}
                if imgs is not None:
                    kwargs["images"] = imgs
                if auds is not None:
                    kwargs["audio"] = auds
                inputs = (
                    proc(**kwargs)
                    if (imgs or auds)
                    else proc.tokenizer(
                        txt, return_tensors="pt", add_special_tokens=False
                    )
                ).to(hf_device)
                gen = hf_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
                hf_expected[name] = gen[0, inputs["input_ids"].shape[1] :].tolist()

        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        llm = LLM(
            model_path=model_dir,
            device=device,
            tensor_parallel_size=1,
            cache_type="static",
            max_batch_size=1,
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            attn_backend="default",
        )
        try:
            for name, conv, imgs, auds in cases:
                if imgs is not None:
                    msg = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": img_path}},
                                conv[0]["content"][1],
                            ],
                        }
                    ]
                elif auds is not None:
                    msg = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio_url", "audio_url": {"url": wav_path}},
                                conv[0]["content"][1],
                            ],
                        }
                    ]
                else:
                    msg = conv
                t0 = time.perf_counter()
                out = llm.chat(messages=[msg])
                dt_ms = (time.perf_counter() - t0) * 1000.0
                inf_ids = list(out[0].outputs[0].token_ids)
                exp_ids = hf_expected[name]
                ok = inf_ids == exp_ids
                print(
                    f"[{'PASS' if ok else 'FAIL'}] {name}: {len(inf_ids)} tok in {dt_ms:.1f} ms | "
                    f"text={out[0].outputs[0].text!r}"
                )
                assert ok, f"Mismatch on {name}:\n  HF ={exp_ids}\n  INF={inf_ids}"
        finally:
            llm.close()


def verify_moe_26b_mini(device: str, tp_sizes: list[int]) -> None:
    from infinilm import _core as infinicore
    from infinilm.cache import StaticKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from safetensors.torch import save_file
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "tie_word_embeddings": True,
            "dtype": "bfloat16",
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 1024,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 1,
                "head_dim": 128,
                "global_head_dim": 256,
                "attention_k_eq_v": True,
                "attention_bias": False,
                "hidden_activation": "gelu_pytorch_tanh",
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 4096,
                "sliding_window": 64,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": True,
                "num_experts": 16,
                "top_k_experts": 4,
                "moe_intermediate_size": 128,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default",
                    },
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional",
                    },
                },
                "dtype": "bfloat16",
                "pad_token_id": 0,
                "eos_token_id": 1,
            },
        }
        torch.manual_seed(42)
        hf_model = Gemma4ForConditionalGeneration(Gemma4Config(**cfg_dict)).to(
            dtype=torch.bfloat16
        )
        with torch.no_grad():
            for p in hf_model.parameters():
                if p.ndim == 1:
                    p.copy_(torch.randn_like(p) * 0.2 + 1.0)
                else:
                    p.copy_(torch.randn_like(p) * 0.05)
            for name, b in hf_model.named_buffers():
                if "layer_scalar" in name:
                    b.fill_(0.85)
            hf_model.tie_weights()

        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)
        sd = {
            k: v.detach().cpu().contiguous()
            for k, v in hf_model.state_dict().items()
            if k != "lm_head.weight"
        }
        save_file(sd, os.path.join(tmpdir, "model.safetensors"))

        hf_dev = "cuda:0" if device == "cuda" else "cpu"
        hf_model = hf_model.to(hf_dev).eval()
        seq = [2, 17, 45, 102, 88, 33]
        expected = []
        cur = torch.tensor([seq], dtype=torch.long, device=hf_dev)
        with torch.no_grad():
            for _ in range(3):
                nxt = int(hf_model(input_ids=cur).logits[0, -1].float().argmax().item())
                expected.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=hf_dev)], dim=1)
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for tp in tp_sizes:
            engine = InferEngine(
                model_path=tmpdir,
                device=infinicore.device(device, 0),
                distributed_config=DistConfig(tp),
                cache_config=StaticKVCacheConfig(1, 256),
                attention_backend="default",
            )
            load_model_state_dict_by_file(engine, tmpdir, dtype=engine.dtype)

            def step(ids_list, pos_list, past_len, total_len):
                out = engine.forward(
                    infinicore.from_torch(torch.tensor([ids_list], dtype=torch.int64)),
                    position_ids=infinicore.from_torch(
                        torch.tensor(pos_list, dtype=torch.int64)
                    ),
                    past_kv_lengths=infinicore.from_torch(
                        torch.tensor([past_len], dtype=torch.int32)
                    ),
                    total_kv_lengths=infinicore.from_torch(
                        torch.tensor([total_len], dtype=torch.int32)
                    ),
                    input_offsets=infinicore.from_torch(
                        torch.tensor([0, len(ids_list)], dtype=torch.int32)
                    ),
                    cu_seqlens=infinicore.from_torch(
                        torch.tensor([0, total_len], dtype=torch.int32)
                    ),
                    temperature=1.0,
                    top_k=1,
                    top_p=1.0,
                )
                return int(out.to_numpy().reshape(-1)[-1])

            got = [step(seq, list(range(len(seq))), 0, len(seq))]
            got.append(step([got[-1]], [len(seq)], len(seq), len(seq) + 1))
            got.append(step([got[-1]], [len(seq) + 1], len(seq) + 1, len(seq) + 2))
            ok = got == expected
            print(
                f"[{'PASS' if ok else 'FAIL'}] moe_26b_mini TP={tp}: got={got}, expected={expected}"
            )
            assert ok, f"MoE TP={tp} mismatch: got={got}, expected={expected}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 InfiniLM parity suite")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to gemma-4-E2B-it"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--tp-sizes",
        type=str,
        default="1",
        help="Comma-separated TP sizes for MoE test",
    )
    args = parser.parse_args()

    if args.model:
        verify_e2b(args.model, args.device)
    tp_list = [int(x) for x in args.tp_sizes.split(",") if x.strip()]
    verify_moe_26b_mini(args.device, tp_list)
    print("ALL_GEMMA4_PARITY_CHECKS_PASSED")


if __name__ == "__main__":
    main()
