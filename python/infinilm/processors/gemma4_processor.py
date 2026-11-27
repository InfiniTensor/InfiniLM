from __future__ import annotations

import json
import os
from typing import Any

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast
from typing_extensions import override

from ..llm.scheduler import SchedulerOutput
from ..llm.static_scheduler import StaticSchedulerOutput
from .basic_llm_processor import BasicLLMProcessor


def _load_audio_waveform(audio_item: Any, target_sr: int = 16000):
    """Load an audio path or pre-loaded numpy array as 1-D float32 waveform."""
    import numpy as np

    if isinstance(audio_item, np.ndarray):
        return audio_item.astype(np.float32, copy=False)
    if isinstance(audio_item, str):
        import soundfile as sf

        data, sr = sf.read(audio_item, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if sr != target_sr:
            # Simple linear resample fallback when librosa/scipy is unavailable
            new_len = int(round(len(data) * float(target_sr) / float(sr)))
            x_old = np.linspace(0.0, 1.0, len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
        return data
    return np.asarray(audio_item, dtype=np.float32)


class Gemma4Processor(BasicLLMProcessor):
    """Processor for Gemma 4 (E2B / E4B / 26B-A4B / 31B) text, image & audio inputs."""

    def __init__(self, model_dir_path: str):
        self.model_dir_path = model_dir_path
        self.processor = None
        self.config_json = {}
        self.model_dtype = None
        config_path = os.path.join(model_dir_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config_json = json.load(f)
            dtype_name = (
                self.config_json.get("torch_dtype")
                or self.config_json.get("dtype")
                or self.config_json.get("text_config", {}).get("torch_dtype")
                or self.config_json.get("text_config", {}).get("dtype")
                or "bfloat16"
            )
            try:
                import torch

                self.model_dtype = getattr(torch, str(dtype_name))
            except Exception:
                self.model_dtype = None

        self.image_token_id = int(self.config_json.get("image_token_id", 258880))
        self.audio_token_id = int(self.config_json.get("audio_token_id", 258881))

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            except Exception:
                tokenizer_file = os.path.join(model_dir_path, "tokenizer.json")
                if not os.path.exists(tokenizer_file):
                    raise
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
                tok_cfg_path = os.path.join(model_dir_path, "tokenizer_config.json")
                if os.path.exists(tok_cfg_path):
                    with open(tok_cfg_path, "r") as f:
                        tok_cfg = json.load(f)
                    for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
                        val = tok_cfg.get(attr)
                        if isinstance(val, dict):
                            val = val.get("content")
                        if val is not None:
                            setattr(self.tokenizer, attr, val)
                chat_template_path = os.path.join(model_dir_path, "chat_template.jinja")
                if os.path.exists(chat_template_path):
                    with open(chat_template_path, "r") as f:
                        self.tokenizer.chat_template = f.read()

    @override
    def __call__(
        self,
        prompt,
        images=None,
        videos=None,
        audios=None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        if not images and not videos and not audios:
            return self.tokenizer(
                prompt, return_tensors=return_tensors, add_special_tokens=False
            )

        if self.processor is None:
            raise RuntimeError(
                "Gemma 4 multimodal processing requires transformers>=5.5 with Gemma4Processor"
            )

        media_kwargs: dict[str, Any] = {}
        if images:
            media_kwargs["images"] = images
        if audios:
            media_kwargs["audio"] = [_load_audio_waveform(a) for a in audios]
        if videos:
            media_kwargs["videos"] = videos

        results = self.processor(
            text=prompt,
            return_tensors=return_tensors or "pt",
            **media_kwargs,
            **kwargs,
        )
        self._pack_gemma4_mm_metadata(results)
        return results

    def _find_token_spans(
        self, token_ids: list[int], target_id: int
    ) -> list[list[int]]:
        spans = []
        i = 0
        n = len(token_ids)
        while i < n:
            if int(token_ids[i]) != target_id:
                i += 1
                continue
            start = i
            while i < n and int(token_ids[i]) == target_id:
                i += 1
            spans.append([start, i])
        return spans

    def _pack_gemma4_mm_metadata(self, processed_inputs: dict) -> None:
        """Convert HF Gemma 4 image & audio tensors into InfiniLM's unified mm lists.

        Convention matching csrc/models/gemma4/gemma4_for_causal_lm.cpp:
          - Image entry: payload [grid_h * grid_w, patch_dim], grid [1, grid_h, grid_w], bound [start, end]
          - Audio entry: payload [valid_frames, mel_dim],      grid [0, valid_frames, mel_dim], bound [start, end]
        """
        import torch

        input_ids = processed_inputs["input_ids"]
        token_ids = (
            input_ids[0].tolist()
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2
            else (
                input_ids.tolist()
                if isinstance(input_ids, torch.Tensor)
                else list(input_ids)
            )
        )

        mm_payloads = []
        mm_grids = []
        mm_bounds = []

        if (
            "pixel_values" in processed_inputs
            and processed_inputs["pixel_values"] is not None
        ):
            pv = processed_inputs["pixel_values"]
            pos_ids = processed_inputs["image_position_ids"]
            img_spans = self._find_token_spans(token_ids, self.image_token_id)
            num_images = pv.shape[0]
            if len(img_spans) != num_images:
                raise RuntimeError(
                    f"Gemma4 image span count ({len(img_spans)}) != pixel_values batch ({num_images})"
                )
            for idx in range(num_images):
                pos = pos_ids[idx]
                valid_mask = (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
                valid_pos = pos[valid_mask]
                grid_w = int(valid_pos[:, 0].max().item()) + 1
                grid_h = int(valid_pos[:, 1].max().item()) + 1
                valid_patches = pv[idx][valid_mask].contiguous()
                mm_payloads.append(valid_patches)
                mm_grids.append(torch.tensor([1, grid_h, grid_w], dtype=torch.int64))
                mm_bounds.append(torch.tensor(img_spans[idx], dtype=torch.int64))

        if (
            "input_features" in processed_inputs
            and processed_inputs["input_features"] is not None
        ):
            feats = processed_inputs["input_features"]
            fmask = processed_inputs.get("input_features_mask")
            aud_spans = self._find_token_spans(token_ids, self.audio_token_id)
            num_audios = feats.shape[0]
            if len(aud_spans) != num_audios:
                raise RuntimeError(
                    f"Gemma4 audio span count ({len(aud_spans)}) != input_features batch ({num_audios})"
                )
            for idx in range(num_audios):
                if fmask is not None:
                    valid_f = feats[idx][fmask[idx].bool()].contiguous()
                else:
                    valid_f = feats[idx].contiguous()
                mm_payloads.append(valid_f)
                mm_grids.append(
                    torch.tensor(
                        [0, int(valid_f.shape[0]), int(valid_f.shape[1])],
                        dtype=torch.int64,
                    )
                )
                mm_bounds.append(torch.tensor(aud_spans[idx], dtype=torch.int64))

        if mm_payloads:
            processed_inputs["_gemma4_mm_payloads"] = mm_payloads
            processed_inputs["_gemma4_mm_grids"] = mm_grids
            processed_inputs["_gemma4_mm_bounds"] = mm_bounds

    @override
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized = []
        for message in conversation:
            content = message.get("content", "")
            if isinstance(content, list):
                has_media = any(
                    item.get("type")
                    in (
                        "image_url",
                        "image",
                        "audio_url",
                        "audio",
                        "video_url",
                        "video",
                    )
                    for item in content
                )
                if not has_media:
                    text = "".join(
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text"
                    )
                    normalized.append({**message, "content": text})
                else:
                    norm_items = []
                    for item in content:
                        t = item.get("type")
                        if t == "text":
                            norm_items.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif t in ("image_url", "image"):
                            norm_items.append({"type": "image"})
                        elif t in ("audio_url", "audio"):
                            norm_items.append({"type": "audio"})
                        elif t in ("video_url", "video"):
                            norm_items.append({"type": "video"})
                        else:
                            norm_items.append(item)
                    normalized.append({**message, "content": norm_items})
            else:
                normalized.append(message)

        owner = self.processor if self.processor is not None else self.tokenizer
        return owner.apply_chat_template(
            conversation=normalized,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    @override
    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
        **kwargs,
    ) -> dict:
        if isinstance(scheduler_output, StaticSchedulerOutput):
            model_inputs = self._build_model_input_from_static_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_gemma4_mm_inputs(model_inputs, scheduler_output)
            return model_inputs
        if isinstance(scheduler_output, SchedulerOutput):
            model_inputs = self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
            self._append_gemma4_mm_inputs(model_inputs, scheduler_output)
            return model_inputs
        raise ValueError(
            "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
        )

    def _append_gemma4_mm_inputs(self, model_inputs: dict, scheduler_output) -> None:
        if not getattr(scheduler_output, "is_prefill", False):
            return

        from .. import _core as infinicore

        payloads = []
        grids = []
        bounds = []
        req_ids = []

        for req_id, req in enumerate(scheduler_output.scheduled_requests):
            proc_in = getattr(req, "processed_inputs", None)
            if not proc_in or "_gemma4_mm_payloads" not in proc_in:
                continue
            num_cached = int(getattr(req, "num_local_cached_tokens", 0) or 0)
            for payload, grid, bound in zip(
                proc_in["_gemma4_mm_payloads"],
                proc_in["_gemma4_mm_grids"],
                proc_in["_gemma4_mm_bounds"],
            ):
                if int(bound[1].item()) <= num_cached:
                    continue
                t_payload = (
                    payload.to(self.model_dtype)
                    if self.model_dtype is not None
                    else payload
                )
                payloads.append(infinicore.from_torch(t_payload.contiguous()))
                grids.append(infinicore.from_torch(grid.contiguous()))
                bounds.append(infinicore.from_torch((bound - num_cached).contiguous()))
                req_ids.append(req_id)

        if payloads:
            model_inputs["pixel_values"] = payloads
            model_inputs["image_grid_thw"] = grids
            model_inputs["image_bound"] = bounds
            model_inputs["image_req_ids"] = req_ids

    @override
    def get_mm_token_index_list(
        self, prompt_token_ids, image_ids=None, video_ids=None, audio_ids=None, **kwargs
    ):
        mm_list = []
        image_ids = image_ids or []
        audio_ids = audio_ids or []
        img_spans = self._find_token_spans(prompt_token_ids, self.image_token_id)
        for span, ident in zip(img_spans, image_ids):
            mm_list.append(
                {"start_index": span[0], "end_index": span[1] - 1, "identifier": ident}
            )
        aud_spans = self._find_token_spans(prompt_token_ids, self.audio_token_id)
        for span, ident in zip(aud_spans, audio_ids):
            mm_list.append(
                {"start_index": span[0], "end_index": span[1] - 1, "identifier": ident}
            )
        mm_list.sort(key=lambda x: x["start_index"])
        return mm_list
