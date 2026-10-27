"""Check checkpoint mapping ownership and file protection without a native SDK."""

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "python/infinilm/_safetensors_mmap.py"
_SPEC = importlib.util.spec_from_file_location("_safetensors_mmap_host", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class SafeTensorsMmapHostTest(unittest.TestCase):
    def test_readonly_views_own_mapping_after_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            header = json.dumps(
                {"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            ).encode()
            content = struct.pack("<Q", len(header)) + header + b"\x80\x3f\x00\x40"
            path.write_bytes(content)
            reader = _MODULE.SafeTensorsMmap(path)
            values = reader.tensor_bytes("weight")
            self.assertFalse(values.flags.writeable)
            with self.assertRaises(ValueError):
                values[0] = 0
            # A private mapping must never write through to the checkpoint.
            reader._mapping[reader._data_start] = 0
            self.assertEqual(path.read_bytes(), content)
            reader.close()
            np.testing.assert_array_equal(values, [0x3F00, 0x4000])
            with self.assertRaises(ValueError):
                reader.tensor_bytes("weight")
            del values


if __name__ == "__main__":
    unittest.main()
