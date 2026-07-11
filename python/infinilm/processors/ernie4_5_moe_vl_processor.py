"""Processor for ERNIE-4.5-VL-28B-A3B (model_type: ernie4_5_moe_vl).

Handles the three input modalities required by the task:
  1. text -> text
  2. image + text -> text
  3. video + text -> text

Inherits from BasicLLMProcessor for the text-only path (build_model_inputs,
tokenization, chat template defaults). Multimodal extensions are layered on top.

NOTE on third-party deps: the tokenizer is loaded via transformers.AutoTokenizer
(a standard component, same as BasicLLMProcessor). The multimodal *adaptation*
logic (patchify, placeholder expansion, 3D/mrope position ids) must be implemented
here and must NOT call into a third-party model's forward/processing internals;
transformers may only be used as a reference in the correctness test.
"""

import json
import os
from typing import Optional

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


def _conversation_is_text_only(conversation) -> bool:
    """Return True if no message contains a non-text content item."""
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                # Anything other than {"type": "text", ...} is multimodal.
                if isinstance(item, dict) and item.get("type", "text") != "text":
                    return False
    return True


# ----------------------------------------------------------------------
# Geometry / timestamp helpers -- byte-for-byte mirrors of the HF processor
# (processing_ernie4_5_vl.py) so our independent pipeline lands on the same
# grid_thw, placeholder counts, and burned-in timestamp pixels.
# ----------------------------------------------------------------------
def _round_by_factor(number, factor):
    return round(number / factor) * factor


def _floor_by_factor(number, factor):
    import math

    return math.floor(number / factor) * factor


def _ceil_by_factor(number, factor):
    import math

    return math.ceil(number / factor) * factor


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Round H/W to a multiple of `factor` while keeping H*W within
    [min_pixels, max_pixels] and the aspect ratio as close as possible.
    Returns (h_bar, w_bar), both multiples of factor."""
    import math

    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        if height > width:
            new_width = max(factor, _round_by_factor(width, factor))
            new_height = _floor_by_factor(new_width * MAX_RATIO, factor)
        else:
            new_height = max(factor, _round_by_factor(height, factor))
            new_width = _floor_by_factor(new_height * MAX_RATIO, factor)
        height, width = new_height, new_width

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _timestamp_converting(t):
    """seconds -> 'HH:MM:SS.ss' (mirror of HF timestamp_converting)."""
    hours = 0
    while t >= 3600:
        hours += 1
        t -= 3600
    mins = 0
    while t >= 60:
        mins += 1
        t -= 60
    return f"{int(hours):02d}:{int(mins):02d}:{t:05.02f}"


@register_processor("ernie4_5_moe_vl")
class Ernie4_5_VLMoeProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        # Tokenizer setup via parent (transformers AutoTokenizer with trust_remote_code).
        # Uses the Ernie4_5_VLTokenizer class declared in tokenizer_config.json's
        # auto_map, defined in processing_ernie4_5_vl.py in the model dir.
        super().__init__(model_dir_path)

        with open(os.path.join(model_dir_path, "config.json")) as f:
            cfg = json.load(f)

        # Model compute dtype (pixel_values must match the vision weights).
        self._config_dtype = cfg.get("torch_dtype", "bfloat16")

        # Special token ids for vision placeholders / span markers.
        self.im_patch_id = cfg.get("im_patch_id", 100295)
        self.image_start_token_id = cfg.get("image_start_token_id", 101304)
        self.image_end_token_id = cfg.get("image_end_token_id", 101305)
        self.video_start_token_id = cfg.get("video_start_token_id", 101306)
        self.video_end_token_id = cfg.get("video_end_token_id", 101307)

        # Vision / resampler geometry.
        vision_cfg = cfg.get("vision_config", {})
        self.patch_size = vision_cfg.get("patch_size", 14)
        self.spatial_merge_size = vision_cfg.get("spatial_merge_size", 2)
        self.temporal_conv_size = cfg.get("temporal_conv_size", 2)
        self.spatial_conv_size = cfg.get("spatial_conv_size", 2)

        # Video frame-sampling + pixel-budget config. Mirrors the HF processor
        # (processing_ernie4_5_vl.py) defaults; read from preprocessor_config.json
        # when present, else fall back to the shipped defaults.
        pp = {}
        pp_path = os.path.join(model_dir_path, "preprocessor_config.json")
        if os.path.exists(pp_path):
            with open(pp_path) as f:
                pp = json.load(f)
        self.video_fps = pp.get("fps", 2)
        self.video_min_frames = pp.get("min_frames", 16)
        self.video_max_frames = pp.get("max_frames", 180)
        self.video_min_pixels = pp.get("video_min_pixels", 234416)
        self.video_max_pixels = pp.get("video_max_pixels", 937664)

    # Placeholder markers the tokenizer chat template emits for each media item
    # (its render_content macro renders " Picture N:<|IMAGE_START|><|image@placeholder
    # |><|IMAGE_END|>" / the video equivalent). __call__ splits the rendered prompt on
    # these and splices the per-image patch-placeholder token run in their place.
    IMAGE_SENTINEL = "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"
    VIDEO_SENTINEL = "<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"

    # ------------------------------------------------------------------
    # Chat template: defer to the tokenizer's template for BOTH text and multimodal.
    # The template's render_content macro turns image/video content items into the
    # markers above (with the "Picture N:" / "Video N:" prefixes ERNIE was trained
    # with), so we pass the conversation through unchanged -- flattening it to a
    # custom sentinel string (the previous approach) bypassed that macro and dropped
    # the "Picture N:" prefix, which corrupted the multimodal prompt. __call__ then
    # expands each marker into image_start + im_patch * N + image_end token ids.
    # tokenize=False: the LLM pipeline tokenizes via __call__.
    # ------------------------------------------------------------------
    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        if _conversation_is_text_only(conversation):
            return super().apply_chat_template(
                conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        return self.tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Text-only entry: defer to BasicLLMProcessor.__call__ unless multimodal.
    # ------------------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        images: Optional[list] = None,
        videos: Optional[list] = None,
        audios: Optional[list] = None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        # Pure text path — same as BasicLLMProcessor (drives the text correctness test).
        if not images and not videos:
            return super().__call__(prompt, return_tensors=return_tensors, **kwargs)

        # Multimodal path: walk the rendered prompt in order, replacing each
        # image/video sentinel with its start + im_patch*N + end run and appending
        # that media's patches + grid row. Ordering media by sentinel position keeps
        # grid_thw / pixel_values in the same order the C++ tower + resampler +
        # merge_vision_embeddings consume them (they scatter vision tokens onto the
        # im_patch positions left-to-right). Image and video both use im_patch_id as
        # the placeholder (HF video_patch_id falls back to image_patch_id).
        import re

        import numpy as np

        images = list(images or [])
        videos = list(videos or [])
        pattern = re.compile(
            "(" + re.escape(self.IMAGE_SENTINEL) + "|" + re.escape(self.VIDEO_SENTINEL) + ")"
        )
        ids = []
        pixel_list = []
        grid_list = []
        img_i = vid_i = 0
        for part in pattern.split(prompt):
            if part == self.IMAGE_SENTINEL:
                patches, grid, count = self._preprocess_one_image(images[img_i])
                img_i += 1
                pixel_list.append(patches)
                grid_list.append(grid)
                ids.append(self.image_start_token_id)
                ids.extend([self.im_patch_id] * count)
                ids.append(self.image_end_token_id)
            elif part == self.VIDEO_SENTINEL:
                patches, grid, count = self._preprocess_one_video(videos[vid_i])
                vid_i += 1
                pixel_list.append(patches)
                grid_list.append(grid)
                ids.append(self.video_start_token_id)
                ids.extend([self.im_patch_id] * count)
                ids.append(self.video_end_token_id)
            elif part:
                ids.extend(self.tokenizer(part, add_special_tokens=False)["input_ids"])

        if img_i != len(images) or vid_i != len(videos):
            raise ValueError(
                f"sentinel/media mismatch: images {img_i}/{len(images)}, "
                f"videos {vid_i}/{len(videos)}"
            )

        return {
            "input_ids": self._wrap_input_ids(ids, return_tensors),
            "pixel_values": np.concatenate(pixel_list, axis=0),
            "grid_thw": np.asarray(grid_list, dtype=np.int64),
        }

    def _preprocess_one_image(self, image_input):
        """Normalize + patchify a single image. Returns (patches [n,3,p,p],
        grid_row [1,h,w], placeholder_count). count = (h*w)//spatial_block is the
        number of vision tokens the resampler emits for a t==1 media, so the C++
        merge_vision_embeddings scatters them 1:1 onto the im_patch positions."""
        block = self.spatial_conv_size * self.spatial_conv_size
        pil = self._load_image(image_input)
        chw = self._normalize_image(pil)
        patches, h, w = self._patchify_frame(chw)
        return patches, [1, h, w], (h * w) // block

    @staticmethod
    def _wrap_input_ids(ids, return_tensors):
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids], dtype=torch.long)
        return ids

    # ------------------------------------------------------------------
    # Multimodal preprocessing.
    # ------------------------------------------------------------------
    # CLIP-style normalization (used by most large vision encoders, including
    # ERNIE-VL's DFNRope ViT). Confirm against the checkpoint's preprocessor_config
    # if accuracy is off.
    _IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    def _patchify_frame(self, np_chw):
        """Take a [3, H, W] float32 array (already normalized) and emit
        patches in spatial-merge-friendly order along with the patch grid.

        Returns: (patches [num_patches, 3, patch, patch], h_grid, w_grid).
        The patches are laid out so every spatial_merge x spatial_merge block
        of consecutive entries forms one 2x2 spatial group — matches the
        resampler's view() contract.
        """
        import numpy as np

        _, H, W = np_chw.shape
        p = self.patch_size
        m = self.spatial_merge_size
        assert H % (p * m) == 0 and W % (p * m) == 0, (
            f"image size {(H, W)} must be a multiple of patch*merge={p * m}"
        )

        h = H // p
        w = W // p
        # [3, h*p, w*p] -> [3, h, p, w, p] -> [h, w, 3, p, p]
        x = np_chw.reshape(3, h, p, w, p).transpose(1, 3, 0, 2, 4)
        # Group neighbouring 2x2 patches: [h/m, m, w/m, m, 3, p, p]
        x = x.reshape(h // m, m, w // m, m, 3, p, p)
        # Reorder so each spatial group is contiguous: [h/m, w/m, m, m, 3, p, p]
        x = x.transpose(0, 2, 1, 3, 4, 5, 6)
        # Flatten: [num_patches, 3, p, p]
        patches = x.reshape(-1, 3, p, p).astype(np.float32, copy=False)
        return patches, h, w

    def _load_image(self, image_input):
        from PIL import Image

        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _normalize_image(self, pil_img):
        """Resize to a multiple of patch*spatial_merge, normalize, return [3,H,W]."""
        import numpy as np
        from PIL import Image

        cell = self.patch_size * self.spatial_merge_size
        W, H = pil_img.size
        # Round up to the next cell boundary; the vision tower handles variable
        # resolution so we don't need a fixed canonical size.
        new_W = max(((W + cell - 1) // cell) * cell, cell)
        new_H = max(((H + cell - 1) // cell) * cell, cell)
        if (new_W, new_H) != (W, H):
            pil_img = pil_img.resize((new_W, new_H), Image.BICUBIC)

        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        mean = np.array(self._IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self._IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def _frame_to_chw(self, pil_img, target_h, target_w):
        """Resize a PIL frame to (target_h, target_w) and CLIP-normalize to
        [3,H,W]. Same normalization as _normalize_image, but the target size comes
        from smart_resize (video) rather than round-up-to-cell."""
        import numpy as np
        from PIL import Image

        pil_img = pil_img.convert("RGB")
        if pil_img.size != (target_w, target_h):
            pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        mean = np.array(self._IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self._IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    _FONT_URL = (
        "https://paddlenlp.bj.bcebos.com/vision-language-models/materials/Roboto-Regular.ttf"
    )

    def _font_path(self):
        """Path to Roboto-Regular.ttf (the font HF burns timestamps with). Cached
        next to this module; downloaded from the same source on first use so the
        rendered glyphs are pixel-identical to HF's render_frame_timestamp."""
        path = os.path.join(os.path.dirname(__file__), "Roboto-Regular.ttf")
        if not os.path.exists(path):
            import requests

            resp = requests.get(self._FONT_URL)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        return path

    def _render_timestamp(self, pil_img, timestamp, font_rate=0.1):
        """Burn 'time: HH:MM:SS.ss' into the top-left corner (black fill, white
        stroke = 10% of font). Mirrors HF render_single_image_with_timestamp and,
        like HF, runs at native resolution *before* smart-resize."""
        from PIL import ImageDraw, ImageFont

        text = "time: " + _timestamp_converting(timestamp)
        draw = ImageDraw.Draw(pil_img)
        width, height = pil_img.size
        font_size = int(min(width, height) * font_rate)
        outline_size = int(font_size * 0.1)
        font = ImageFont.truetype(self._font_path(), font_size)
        draw.text(
            (0, 0),
            text,
            font=font,
            fill=(0, 0, 0),
            stroke_width=outline_size,
            stroke_fill=(255, 255, 255),
        )
        return pil_img

    def _get_frame_indices(self, vlen, target_frames, target_fps, input_fps):
        """Mirror of HF get_frame_indices for frames_sample='leading'.

        target_frames>0: split [0,vlen) into target_frames intervals, take each
        interval's left edge. Else (fps mode): sample at 1/target_fps s and round
        to the nearest source frame index."""
        import numpy as np

        if target_frames > 0:
            acc = min(target_frames, vlen)
            intervals = np.linspace(0, vlen, acc + 1).astype(int)
            return [int(intervals[i]) for i in range(acc)]  # leading = interval left edge
        delta = 1.0 / target_fps
        duration = float(vlen) / input_fps
        frame_seconds = np.arange(0, duration, delta)
        idx = np.around(frame_seconds * input_fps).astype(int)
        return [int(e) for e in idx if e < vlen]

    def _decode_and_sample_frames(self, video_input):
        """decord decode + HF-matching frame sampling + timestamp burn-in + even
        pad. Returns RGB PIL frames at the video's native resolution."""
        from decord import VideoReader
        from PIL import Image

        if not isinstance(video_input, str):
            raise ValueError(
                f"video input must be a file path str, got {type(video_input)}"
            )
        vr = VideoReader(video_input, num_threads=1)
        vlen = len(vr)
        input_fps = float(vr.get_avg_fps())
        duration = vlen / input_fps

        # _set_video_frame_args: target_frames=-1 -> fps path, clamped into
        # [min_frames, max_frames] by switching to a fixed target_frames count.
        fps = self.video_fps
        target_frames = -1
        frames_to_extract = int(duration * fps)
        if self.video_min_frames > 0 and frames_to_extract < self.video_min_frames:
            target_frames = self.video_min_frames
            fps = -1
        elif self.video_max_frames > 0 and frames_to_extract > self.video_max_frames:
            target_frames = self.video_max_frames
            fps = -1

        frame_indices = self._get_frame_indices(vlen, target_frames, fps, input_fps)

        frames = [Image.fromarray(vr[fi].asnumpy(), "RGB") for fi in frame_indices]
        # timestamp(sec) = frame_idx * duration / num_of_frame = frame_idx / input_fps
        timestamps = [fi * duration / vlen for fi in frame_indices]
        frames = [self._render_timestamp(f, ts) for f, ts in zip(frames, timestamps)]
        # Resampler temporal merge is 2:1, so frame count must be even (HF pads the
        # last frame when odd).
        if len(frames) % 2 != 0:
            frames.append(frames[-1].copy())
        return frames

    def _preprocess_one_video(self, video_input):
        """Decode -> sample -> burn timestamps -> smart-resize -> patchify a single
        video. Returns (patches [t*ph*pw,3,p,p], grid_row [t,ph,pw], count).

        count = t*ph*pw // (spatial_block * temporal_conv_size) = t_eff*gh*gw, the
        number of vision tokens the resampler emits after spatial + temporal merge."""
        import numpy as np

        frames = self._decode_and_sample_frames(video_input)
        t = len(frames)
        w0, h0 = frames[0].size  # PIL size is (W, H)
        factor = self.patch_size * self.spatial_merge_size
        rh, rw = smart_resize(h0, w0, factor, self.video_min_pixels, self.video_max_pixels)
        ph, pw = rh // self.patch_size, rw // self.patch_size

        patch_list = []
        for f in frames:
            chw = self._frame_to_chw(f, rh, rw)
            patches, _, _ = self._patchify_frame(chw)  # h==ph, w==pw by construction
            patch_list.append(patches)

        pixel_values = np.concatenate(patch_list, axis=0)
        block = self.spatial_conv_size * self.spatial_conv_size
        count = (t * ph * pw) // (block * self.temporal_conv_size)
        return pixel_values, [t, ph, pw], count

    def _build_3d_position_ids(self, input_ids, grid_thw):
        """Qwen2-VL-style get_rope_index -> [3][seq] = (time, height, width).

        Text tokens advance sequentially on all three axes. Each im_patch run gets
        2D (height,width) grid positions offset by the running start; the next text
        token resumes from max(position)+1. The merged grid uses spatial_conv_size
        (the resampler's spatial merge): hh=h//s, ww=w//s.
        VERIFY: match HF modeling_ernie4_5_vl.get_rope_index (axis order + the
        position offset accounting after each image span).
        """
        s = self.spatial_conv_size
        grids = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
        tpos, hpos, wpos = [], [], []
        st = 0
        img_idx = 0
        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i] == self.im_patch_id:
                t, h, w = grids[img_idx]
                img_idx += 1
                hh, ww = int(h) // s, int(w) // s
                # Temporal merge: the resampler downsamples time by temporal_conv_size
                # (video), so the placeholder run is t_eff*hh*ww long, not t*hh*ww.
                # Mirrors HF _compute_3d_positions: t_eff = t//temporal_conv if t!=1
                # else 1. Image (t==1) is unchanged.
                t_eff = int(t) // self.temporal_conv_size if int(t) != 1 else 1
                count = t_eff * hh * ww
                for idx in range(count):
                    ti = idx // (hh * ww)
                    rem = idx % (hh * ww)
                    hi = rem // ww
                    wi = rem % ww
                    tpos.append(st + ti)
                    hpos.append(st + hi)
                    wpos.append(st + wi)
                # Next start = max coord + 1 = st + max(t_eff, hh, ww)
                # (HF: cur_position = np.max(pos_ids) + 1).
                st = st + max(t_eff, hh, ww)
                i += count
            else:
                tpos.append(st)
                hpos.append(st)
                wpos.append(st)
                st += 1
                i += 1
        return [tpos, hpos, wpos]

    def build_model_inputs(self, scheduler_output, temperature=1.0, top_p=0.8, top_k=1):
        """Inject multimodal tensors + 3D mrope positions onto the base text inputs.

        Multimodal data lives on the request's processed_inputs (from __call__).
        pixel_values/tgt_sizes are sent only during prefill (vision is cached after);
        position_ids are replaced with the [3, seq] mrope layout for the tokens
        computed this step. Text-only requests fall through to the base unchanged.
        """
        base = super().build_model_inputs(scheduler_output, temperature, top_p, top_k)

        reqs = getattr(scheduler_output, "scheduled_requests", None)
        if not reqs:
            return base
        req = reqs[0]
        pi = getattr(req, "processed_inputs", None)
        if not pi or pi.get("pixel_values") is None:
            return base

        import infinicore
        import numpy as np

        grid_thw = np.asarray(pi["grid_thw"]).astype(np.int64)
        pos3d = self._build_3d_position_ids(req.get_all_token_ids(), grid_thw)

        if getattr(scheduler_output, "is_prefill", True):
            prefix = getattr(scheduler_output, "prefix_hit_len", 0) or 0
            end = len(req.get_input_tokens())
            pos_slice = [row[prefix:end] for row in pos3d]

            # Flatten patches to [num_patches, C*p*p]; the C++ patch_embed views it
            # back. Build in model dtype so it matches the vision weights. Use
            # from_numpy (not from_list(.tolist())): a video has ~19200*588 = 11.3M
            # elements, and materializing that as a nested Python list is slow and
            # memory-heavy for no benefit (from_list just rebuilds a numpy array).
            pv2d = np.ascontiguousarray(pi["pixel_values"]).reshape(pi["pixel_values"].shape[0], -1)
            base["pixel_values"] = infinicore.from_numpy(pv2d, dtype=self._infini_dtype())
            base["tgt_sizes"] = infinicore.from_list(grid_thw.tolist(), dtype=infinicore.int64)
        else:
            pos = req.get_total_length() - 1
            pos_slice = [[row[pos]] for row in pos3d]

        base["position_ids"] = infinicore.from_list(pos_slice, dtype=infinicore.int64)

        return base

    def _infini_dtype(self):
        """Model compute dtype for pixel_values (must match the vision weights)."""
        import infinicore

        name = (getattr(self, "_config_dtype", "bfloat16") or "bfloat16").lower()
        return {
            "bfloat16": infinicore.bfloat16,
            "float16": infinicore.float16,
            "float32": infinicore.float32,
        }.get(name, infinicore.bfloat16)
