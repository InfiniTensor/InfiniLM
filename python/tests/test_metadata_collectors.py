"""Tests for server metadata collectors."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from infinilm.server.metadata_collectors import (
    collect_build_info,
    collect_config,
    collect_config_env,
    collect_runtime_env,
)


class TestCollectConfigEnv(unittest.TestCase):
    def test_infinilm_env_keys(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "INFINI_PREFILL_NATIVE_CG": "1",
                "INFINI_COMPILE_MAX_SEQ": "8448",
                "HPCC_VISIBLE_DEVICES": "2",
                "UNRELATED": "x",
            },
            clear=False,
        ):
            env = collect_config_env()
        self.assertEqual(env["INFINI_PREFILL_NATIVE_CG"], "1")
        self.assertEqual(env["INFINI_COMPILE_MAX_SEQ"], "8448")
        self.assertEqual(env["HPCC_VISIBLE_DEVICES"], "2")
        self.assertNotIn("UNRELATED", env)

    def test_collect_config_groups_startup_and_env(self) -> None:
        startup = {"enable_graph": True, "num_blocks": 512}
        with mock.patch.dict(os.environ, {"INFINI_V1_SCHEDULER": "1"}, clear=False):
            cfg = collect_config(startup)
        self.assertEqual(cfg["startup"], startup)
        self.assertEqual(cfg["env"]["INFINI_V1_SCHEDULER"], "1")


class TestCollectBuildInfo(unittest.TestCase):
    def test_env_shas(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "IL_SHA": "abc1234",
                "IC_SHA": "def5678",
                "IO_SHA": "ghi9012",
                "IMAGE_TAG": "infinilm-svc:test",
                "BUILD_TS": "20260701",
            },
            clear=False,
        ):
            info = collect_build_info()
        self.assertEqual(info["il_sha"], "abc1234")
        self.assertEqual(info["ic_sha"], "def5678")
        self.assertEqual(info["io_sha"], "ghi9012")

    def test_parse_shas_from_image_tag(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"IMAGE_TAG": "infinilm-svc:metax-hpcc-ai3107-c73618c-56ef9cad-20260624"},
            clear=False,
        ):
            info = collect_build_info()
        self.assertEqual(info["il_sha"], "c73618c")
        self.assertEqual(info["ic_sha"], "56ef9cad")
        self.assertEqual(info["build_ts"], "20260624")


class TestCollectRuntimeEnv(unittest.TestCase):
    def test_os_probe_from_os_release(self) -> None:
        content = 'ID="kylin"\nVERSION_ID="V10"\n'
        with tempfile.TemporaryDirectory() as tmp:
            os_release = os.path.join(tmp, "os-release")
            with open(os_release, "w", encoding="utf-8") as fh:
                fh.write(content)
            with mock.patch(
                "infinilm.server.metadata_collectors._read_os_release",
                return_value={"ID": "kylin", "VERSION_ID": "V10"},
            ):
                with mock.patch(
                    "infinilm.server.metadata_collectors._probe_cpu",
                    return_value={"cpu_model": "Test CPU", "cpu_count": "4"},
                ):
                    with mock.patch(
                        "infinilm.server.metadata_collectors._probe_gpu",
                        return_value={},
                    ):
                        env = collect_runtime_env()
        self.assertEqual(env["os_id"], "kylin")
        self.assertEqual(env["os_version"], "V10")
        self.assertEqual(env["cpu_model"], "Test CPU")
        self.assertNotIn("INFINI_PREFILL_NATIVE_CG", env)


if __name__ == "__main__":
    unittest.main()
