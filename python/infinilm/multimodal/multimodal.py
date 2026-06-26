from typing import List, Union
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
