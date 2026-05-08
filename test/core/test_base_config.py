import sys
from unittest.mock import patch

from infinilm.base_config import BaseConfig


def test_ascend_config_accepts_paged_attention_defaults():
    argv = [
        "prog",
        "--model",
        "/tmp/model",
        "--device",
        "ascend",
        "--enable-paged-attn",
    ]

    with patch.object(sys, "argv", argv):
        cfg = BaseConfig()

    assert cfg.attn == "flash-attn"
    assert cfg.enable_paged_attn is True
