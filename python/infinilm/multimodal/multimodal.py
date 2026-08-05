import base64
import binascii
import hashlib
from io import BytesIO
from typing import List, Union

from PIL import Image


def _load_image(image_url):
    if isinstance(image_url, str) and image_url.startswith("data:image/"):
        try:
            _, encoded = image_url.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data image URL.") from exc

        try:
            image_data = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("Invalid base64 image data.") from exc

        image_id = f"data:image:{hashlib.sha256(image_data).hexdigest()}"
        return Image.open(BytesIO(image_data)).convert("RGB"), image_id

    return Image.open(image_url).convert("RGB"), image_url


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
                image, image_id = _load_image(item["image_url"]["url"])
                images.append(image)
                image_urls.append(image_id)
            elif item.get("type") == "video_url":
                video = item["video_url"]["url"]
                videos.append(video)
                if isinstance(video, str):
                    video_urls.append(video)
                else:
                    video_urls.append(
                        f"predecoded_video:{len(video_urls)}:{len(video)}"
                    )
            else:  # TODO support audio
                raise NotImplementedError("Only image/video input is supported for now")

    return {
        "images": images,
        "image_urls": image_urls,
        "videos": videos,
        "video_urls": video_urls,
        "audios": audios,
        "audio_urls": audio_urls,
    }
