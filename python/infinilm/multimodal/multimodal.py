import os
from typing import List, Union
from urllib.parse import urlparse

import numpy as np
from PIL import Image


def has_multimodal_inputs(messages: Union[List[dict], dict]) -> bool:
    """Check if the input messages contain any multimodal inputs."""
    if isinstance(messages, dict):
        messages = [messages]

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            return False

        for item in content:
            if item.get("type") in ["image_url", "video_url", "audio_url"]:
                return True

    return False


def _resolve_local_path(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return os.path.expanduser(parsed.path)
    if parsed.scheme in ("", None):
        return os.path.expanduser(url)
    raise NotImplementedError(f"Unsupported multimodal URL scheme: {parsed.scheme}")


def _get_media_url(item: dict, key: str) -> str:
    payload = item.get(key)
    if isinstance(payload, dict):
        payload = payload.get("url")
    if not isinstance(payload, str) or payload == "":
        raise ValueError(f"Missing {key}.url in multimodal input")
    return payload


def _sample_frame_indices(
    frame_count: int,
    fps: float,
    *,
    target_frames: int = -1,
    target_fps: float = 2.0,
    min_frames: int = 16,
    max_frames: int = 180,
    frames_sample: str = "leading",
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video contains no frames")
    if fps <= 0:
        fps = 1.0

    if target_frames <= 0:
        if target_fps <= 0:
            raise ValueError("Either target_frames or target_fps must be positive")
        duration = frame_count / fps
        frame_target = int(duration * target_fps)
        if min_frames > 0 and frame_target < min_frames:
            target_frames = min_frames
        elif max_frames > 0 and frame_target > max_frames:
            target_frames = max_frames
        else:
            seconds = np.arange(0, duration, 1.0 / target_fps)
            indices = np.around(seconds * fps).astype(np.int64).tolist()
            return [idx for idx in indices if 0 <= idx < frame_count] or [0]

    samples = min(max(target_frames, 1), frame_count)
    intervals = np.linspace(0, frame_count, samples + 1).astype(np.int64)
    indices = []
    for start, end in zip(intervals[:-1], intervals[1:]):
        end = max(start, end - 1)
        if frames_sample == "middle":
            indices.append(int((start + end) // 2))
        else:
            indices.append(int(start))
    return indices


def load_video(
    url: str,
    *,
    target_frames: int = -1,
    target_fps: float = 2.0,
    min_frames: int = 16,
    max_frames: int = 180,
    frames_sample: str = "leading",
):
    try:
        import decord
    except ImportError as exc:
        raise ImportError("Video input requires decord to be installed") from exc

    path = _resolve_local_path(url)
    reader = decord.VideoReader(path, num_threads=1)
    frame_count = len(reader)
    fps = float(reader.get_avg_fps() or 1.0)
    frame_indices = _sample_frame_indices(
        frame_count,
        fps,
        target_frames=target_frames,
        target_fps=target_fps,
        min_frames=min_frames,
        max_frames=max_frames,
        frames_sample=frames_sample,
    )
    frames = reader.get_batch(frame_indices).asnumpy()
    if frames.shape[0] % 2 != 0:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
        frame_indices.append(frame_indices[-1])
    metadata = {
        "fps": fps,
        "duration": frame_count / fps,
        "num_of_frame": frame_count,
        "total_num_frames": frame_count,
        "frames_indices": frame_indices,
        "video_backend": "decord",
        "do_sample_frames": False,
    }
    return frames, metadata


def resolve_multimodal_inputs(messages: Union[List[dict], dict]):
    """Get images, videos, audios from the messages."""
    if isinstance(messages, dict):
        messages = [messages]

    images = []
    image_urls = []
    videos = []
    video_urls = []
    audios = []
    audio_urls = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") == "text":
                pass
            elif item.get("type") == "image_url":
                image_url = _get_media_url(item, "image_url")
                images.append(Image.open(_resolve_local_path(image_url)))
                image_urls.append(image_url)
            elif item.get("type") == "video_url":
                payload = item.get("video_url")
                video = payload.get("url") if isinstance(payload, dict) else payload
                if isinstance(video, str):
                    video_args = payload if isinstance(payload, dict) else {}

                    def pick_arg(key, default):
                        return item.get(key, video_args.get(key, default))

                    videos.append(
                        load_video(
                            video,
                            target_frames=int(pick_arg("target_frames", -1)),
                            target_fps=float(pick_arg("fps", 2.0)),
                            min_frames=int(pick_arg("min_frames", 16)),
                            max_frames=int(pick_arg("max_frames", 180)),
                            frames_sample=str(pick_arg("frames_sample", "leading")),
                        )
                    )
                    video_urls.append(video)
                else:
                    videos.append(video)
                    video_urls.append(
                        f"predecoded_video:{len(video_urls)}:{len(video)}"
                    )
            else:  # TODO support audio
                raise NotImplementedError(
                    "Only image and video inputs are supported for now"
                )

    return {
        "images": images,
        "image_urls": image_urls,
        "videos": videos,
        "video_urls": video_urls,
        "audios": audios,
        "audio_urls": audio_urls,
    }
