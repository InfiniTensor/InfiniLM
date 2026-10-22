"""Run actual EngineConfig validation without loading the native package."""
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import unittest

ROOT = Path(__file__).resolve().parents[2]
module = ast.parse((ROOT / "python/infinilm/config/engine_config.py").read_text())
module.body = [node for node in module.body if isinstance(node, ast.ClassDef)]
namespace = dict(dataclass=dataclass, Optional=Optional, KVTransferConfig=object)
exec(compile(module, "engine_config.py", "exec"), namespace)
EngineConfig = namespace["EngineConfig"]


class PagedGraphConfigTest(unittest.TestCase):
    def test_defaults_and_explicit_empty(self):
        self.assertIsNone(EngineConfig("model").graph_capture_profiles)
        self.assertFalse(EngineConfig("model").graph_metadata_upload)
        self.assertEqual(EngineConfig("model", graph_capture_profiles=[]).graph_capture_profiles, [])

    def test_canonical_order_and_deduplication(self):
        config = EngineConfig("model", graph_capture_profiles=[(16,4096),[1,2048],(16,4096),(8,0)])
        self.assertEqual(config.graph_capture_profiles, [(1,2048),(8,0),(16,4096)])

    def test_invalid_profiles(self):
        for profile in [[(0,2048)],[(17,2048)],[(1,-1)],[(True,2048)],[(1,2.5)],[(1,)], ["bad"]]:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                EngineConfig("model", graph_capture_profiles=profile)
        with self.assertRaises(ValueError):
            EngineConfig("model", cache_type="static", graph_capture_profiles=[])

    def test_new_options_do_not_shift_existing_fields(self):
        fields = list(EngineConfig.__dataclass_fields__)
        self.assertEqual(fields[-2:], ["graph_capture_profiles", "graph_metadata_upload"])


if __name__ == "__main__":
    unittest.main()
