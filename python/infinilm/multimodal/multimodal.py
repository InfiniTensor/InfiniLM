from typing import List, Union
from PIL import Image


def resolve_multimodal_inputs(messages: Union[List[dict], dict]):
    """Get images, videos, audios from the messages."""
    if isinstance(messages, dict):
        messages = [messages]

    images = []
    videos = []
    audios = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") == "image":
                # TODO support other image url formats
                images.append(Image.open(item["image_url"]))

            else:  # TODO support video/audio
                raise NotImplementedError("Only image input is supported for now")

    return images, videos, audios
