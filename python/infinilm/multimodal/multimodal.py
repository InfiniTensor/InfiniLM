from typing import List, Optional, Union
from PIL import Image
import xxhash


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
                # TODO support other image url formats
                images.append(Image.open(item["image_url"]["url"]))
                image_urls.append(item["image_url"]["url"])

            else:  # TODO support video/audio
                raise NotImplementedError("Only image input is supported for now")

    return {
        "images": images,
        "image_urls": image_urls,
        "videos": videos,
        "video_urls": video_urls,
        "audios": audios,
        "audio_urls": audio_urls,
    }
