"""Dense checkpoints borrow storage or use qualified Ops CPU float conversion.

These are real native Tensor / mmap ownership tests, not model qualification.
Transformation branches remain explicit until their own replacements pass.
"""

import builtins
import ctypes
import gc
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from infinilm import _core as infinicore
from infinilm import modeling_utils as loading


_MISSING_METADATA = object()


def _write_shard(path, entries, metadata=_MISSING_METADATA):
    header, payload = {}, bytearray()
    if metadata is not _MISSING_METADATA:
        header["__metadata__"] = metadata
    for name, dtype, shape, data in entries:
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class RawCheckpointLoadingTest(unittest.TestCase):
    def test_mamba_raw_remap_cross_shard_real_head_and_owner(self):
        embed_name = "backbone.embeddings.weight"
        source_name = "backbone.layers.0.mixer.conv1d.weight"
        native_name = "backbone.layers.0.mixer.conv1d_weight"
        embed_bits, head_bits = b"\x80\x3f", b"\x00\x40"
        original_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Mamba raw checkpoint imported Torch")
            return original_import(name, *args, **kwargs)

        for declared_tie, real_head, local_head in (
                (False, False, False), (True, False, True),
                (True, True, True), (True, True, False)):
            with self.subTest(tie=declared_tie, real_head=real_head,
                              local_head=local_head), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = {"model_type": "mamba", "tie_word_embeddings": declared_tie}
                (root / "config.json").write_text(json.dumps(config))
                _write_shard(root / "a.safetensors", [
                    (embed_name, "BF16", [1], embed_bits),
                    (source_name, "BF16", [1], b"\x40\x40"),
                ])
                second = [("middle.weight", "BF16", [1], b"\x80\x40")]
                if real_head:
                    second.append(("lm_head.weight", "BF16", [1], head_bits))
                _write_shard(root / "b.safetensors", second)
                _write_shard(root / "c.safetensors", [
                    ("tail.weight", "BF16", [1], b"\xc0\x40")])
                wanted = {embed_name, native_name, "middle.weight", "tail.weight"}
                if local_head:
                    wanted.add("lm_head.weight")
                owners, seen = [], []
                test = self

                class Consumer:
                    hf_config = config
                    distributed_config = SimpleNamespace(pp_size=2)

                    def state_dict_keyname(self):
                        return list(wanted)

                    def load_state_dict(self, tensors, strict):
                        test.assertFalse(strict)
                        test.assertNotIn(source_name, tensors)
                        for key, tensor in tensors.items():
                            test.assertIsInstance(tensor, infinicore.Tensor)
                            test.assertEqual(tensor.device.type, "cpu")
                            seen.append(key)
                        if embed_name in tensors:
                            test.assertFalse(tensors[embed_name]._owner.flags.writeable)
                            owners.append(weakref.ref(tensors[embed_name]._owner))
                        if "lm_head.weight" in tensors:
                            expected = head_bits if real_head else embed_bits
                            test.assertEqual(ctypes.string_at(
                                tensors["lm_head.weight"].data_ptr(), 2), expected)

                    def process_weights_after_loading(self):
                        test.assertCountEqual(seen, wanted)
                        test.assertIsNone(owners[0]())

                real_reader = loading._load_checkpoint_shard

                def observe_next_open(path, *args, **kwargs):
                    test.assertTrue(kwargs["raw"])
                    if Path(path).name == "b.safetensors":
                        test.assertEqual(owners[0]() is not None,
                                         declared_tie and local_head)
                    if Path(path).name == "c.safetensors":
                        test.assertEqual(owners[0]() is not None,
                                         declared_tie and local_head and not real_head)
                    return real_reader(path, *args, **kwargs)

                with patch("builtins.__import__", side_effect=no_torch), \
                     patch.object(loading, "load_state_dict", side_effect=AssertionError(
                         "Torch checkpoint bridge used")), \
                     patch.object(loading, "_load_checkpoint_shard", side_effect=observe_next_open):
                    loading.load_model_state_dict_by_file(
                        Consumer(), str(root), dtype=infinicore.bfloat16)

    def test_mamba_raw_remap_collision_rejected_before_dispatch(self):
        prefix = "backbone.layers.0.mixer."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "mamba"}))
            _write_shard(root / "model.safetensors", [
                (prefix + "conv1d.weight", "BF16", [1], b"\x80\x3f"),
                (prefix + "conv1d_weight", "BF16", [1], b"\x00\x40"),
            ])
            consumer = SimpleNamespace(
                hf_config={"model_type": "mamba"},
                state_dict_keyname=lambda: [prefix + "conv1d_weight"],
                load_state_dict=lambda *_args, **_kwargs: self.fail(
                    "collision dispatched to model"),
            )
            with patch.object(loading, "load_state_dict", side_effect=AssertionError(
                    "Torch checkpoint bridge used")):
                with self.assertRaisesRegex(ValueError, "Mamba weight remap collision"):
                    loading.load_model_state_dict_by_file(
                        consumer, str(root), dtype=infinicore.bfloat16)

    def test_mamba_default_remap_retains_bin_tie_and_legacy_collision_semantics(self):
        first, second, embedding = object(), object(), object()
        result = loading._remap_mamba({
            "x.mixer.conv1d.weight": first,
            "x.mixer.conv1d_weight": second,
            "backbone.embeddings.weight": embedding,
        }, config={"tie_word_embeddings": True})
        self.assertIs(result["x.mixer.conv1d_weight"], second)
        self.assertIs(result["lm_head.weight"], embedding)

    def test_embedding_owner_lives_only_until_required_tie_or_copy_finishes(self):
        # Observe ownership at the next shard's open, not just after the whole
        # loader returns. Retaining an embedding can otherwise retain its mmap.
        for declared_tie, actual_head, local_head in (
                (False, False, False), (True, False, False),
                (True, True, True), (True, False, True)):
            with self.subTest(tie=declared_tie, head=actual_head, local=local_head), \
                 tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = {"model_type": "qwen2", "tie_word_embeddings": declared_tie}
                (root / "config.json").write_text(json.dumps(config))
                embed_name = "model.embed_tokens.weight"
                _write_shard(root / "a.safetensors", [(embed_name, "BF16", [1], b"\x80\x3f")])
                middle = [("middle.weight", "BF16", [1], b"\x00\x40")]
                if actual_head:
                    middle.append(("lm_head.weight", "BF16", [1], b"\x40\x40"))
                _write_shard(root / "b.safetensors", middle)
                _write_shard(root / "c.safetensors", [("tail.weight", "BF16", [1], b"\x80\x40")])
                wanted = [embed_name, "middle.weight", "tail.weight"]
                if local_head:
                    wanted.append("lm_head.weight")
                owners, seen = [], []
                test = self

                class Consumer:
                    hf_config = config
                    distributed_config = SimpleNamespace(pp_size=2)

                    def state_dict_keyname(self):
                        return wanted

                    def load_state_dict(self, tensors, strict):
                        test.assertFalse(strict)
                        seen.extend(tensors)
                        if embed_name in tensors:
                            owners.append(weakref.ref(tensors[embed_name]._owner))
                        if "lm_head.weight" in tensors:
                            expected = b"\x40\x40" if actual_head else b"\x80\x3f"
                            test.assertEqual(ctypes.string_at(tensors["lm_head.weight"].data_ptr(), 2), expected)

                    def process_weights_after_loading(self):
                        test.assertCountEqual(seen, wanted)
                        test.assertIsNone(owners[0]())

                original_reader = loading._load_checkpoint_shard

                def observe_next_open(path, *args, **kwargs):
                    name = Path(path).name
                    if name != "a.safetensors":
                        needed = declared_tie and local_head and (
                            name == "b.safetensors" or not actual_head)
                        test.assertEqual(owners[0]() is not None, needed,
                                         f"embedding ownership before {name}")
                    return original_reader(path, *args, **kwargs)

                with patch.object(loading, "_load_checkpoint_shard", side_effect=observe_next_open), \
                     patch.object(loading, "load_state_dict", side_effect=AssertionError("Torch bridge used")):
                    loading.load_model_state_dict_by_file(
                        Consumer(), str(root), dtype=infinicore.bfloat16)

    def test_raw_metadata_distinguishes_missing_empty_and_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.safetensors"
            entry = [("weight", "BF16", [1], b"\x80\x3f")]
            with patch.object(loading, "load_state_dict", side_effect=AssertionError(
                    "valid raw metadata entered Torch bridge")):
                _write_shard(path, entry)
                self.assertEqual(
                    loading._load_checkpoint_shard(
                        str(path), infinicore.bfloat16, (), raw=True)["weight"].dtype,
                    infinicore.bfloat16)
                _write_shard(path, entry, metadata={"format": "pt"})
                self.assertEqual(
                    loading._load_checkpoint_shard(
                        str(path), infinicore.bfloat16, (), raw=True)["weight"].dtype,
                    infinicore.bfloat16)
                _write_shard(path, entry, metadata={})
                with self.assertRaisesRegex(OSError, "Invalid safetensors metadata"):
                    loading._load_checkpoint_shard(
                        str(path), infinicore.bfloat16, (), raw=True)
                _write_shard(path, entry, metadata=None)
                with self.assertRaisesRegex(ValueError, "metadata must map"):
                    loading._load_checkpoint_shard(
                        str(path), infinicore.bfloat16, (), raw=True)

    def test_untransformed_families_use_real_borrowed_views(self):
        original_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("untransformed checkpoint imported Torch")
            return original_import(name, *args, **kwargs)

        expected = struct.pack("<4H", 0x3f80, 0xc000, 0x4000, 0x0000)
        bias = struct.pack("<f", 0.125)
        integer = struct.pack("<i", 7)
        for family in ("qwen2", "qwen3", "llama", "internlm3", "mistral", "minicpm"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = {"model_type": family, "tie_word_embeddings": True}
                (root / "config.json").write_text(json.dumps(config))
                _write_shard(root / "part-a.safetensors", [
                    ("model.embed_tokens.weight", "BF16", [2, 2], expected),
                    ("router.e_score_correction_bias", "F32", [1], bias),
                    ("indices", "I32", [1], integer),
                ])
                _write_shard(root / "part-b.safetensors", [
                    ("other_stage.weight", "BF16", [2, 2], expected),
                ])
                (root / "model.safetensors.index.json").write_text(json.dumps({
                    "weight_map": {
                        "model.embed_tokens.weight": "part-a.safetensors",
                        "router.e_score_correction_bias": "part-a.safetensors",
                        "indices": "part-a.safetensors",
                        "other_stage.weight": "part-b.safetensors",
                    }}))
                wanted = {"model.embed_tokens.weight": (expected, infinicore.bfloat16),
                          "lm_head.weight": (expected, infinicore.bfloat16),
                          "router.e_score_correction_bias": (bias, infinicore.float32),
                          "indices": (integer, infinicore.int32)}
                seen, owners, pointers = [], [], {}
                test = self

                class Consumer:
                    hf_config = config
                    distributed_config = SimpleNamespace(pp_size=2)

                    def state_dict_keyname(self):
                        return list(wanted)

                    def load_state_dict(self, tensors, strict):
                        test.assertFalse(strict)
                        test.assertNotIn("other_stage.weight", tensors)
                        for name, tensor in tensors.items():
                            bits, dtype = wanted[name]
                            test.assertIsInstance(tensor, infinicore.Tensor)
                            test.assertEqual(tensor.device.type, "cpu")
                            test.assertEqual(tensor.dtype, dtype)
                            test.assertEqual(ctypes.string_at(tensor.data_ptr(), len(bits)), bits)
                            test.assertFalse(tensor._owner.flags.writeable)
                            pointers[name] = tensor.data_ptr()
                            owners.append(weakref.ref(tensor._owner))
                            seen.append(name)

                    def process_weights_after_loading(self):
                        test.assertCountEqual(seen, wanted)
                        test.assertEqual(pointers["model.embed_tokens.weight"],
                                         pointers["lm_head.weight"])

                with patch("builtins.__import__", side_effect=no_torch), patch.object(
                    loading, "load_state_dict", side_effect=AssertionError("Torch bridge used")
                ):
                    loading.load_model_state_dict_by_file(
                        Consumer(), str(root), dtype=infinicore.bfloat16)
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))

    def test_transformations_keep_their_existing_bridge(self):
        class BridgeRequired(Exception):
            pass

        cases = (
            {"model_type": "llama", "quantization_config": {"quant_method": "awq"}},
            {"model_type": "minicpm", "scale_emb": 2.0},
            {"model_type": "gpt2"},
        )
        for config in cases:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(config))
                _write_shard(root / "model.safetensors", [
                    ("weight", "BF16", [1], b"\x80\x3f"),
                ])
                consumer = SimpleNamespace(hf_config=config, state_dict_keyname=lambda: ["weight"])
                real_reader = loading._load_checkpoint_shard

                def check_bridge(*args, **kwargs):
                    self.assertFalse(kwargs["raw"])
                    with patch.object(loading, "_require_torch_checkpoint", return_value=object()) as required, \
                         patch.object(loading.infinicore.utils, "to_torch_dtype", return_value="bf16"), \
                         patch.object(loading, "load_state_dict", side_effect=BridgeRequired) as bridge:
                        try:
                            return real_reader(*args, **kwargs)
                        finally:
                            required.assert_called_once_with("unqualified safetensors checkpoint fallback")
                            bridge.assert_called_once()

                with patch.object(loading, "_load_checkpoint_shard", side_effect=check_bridge):
                    with self.assertRaises(BridgeRequired):
                        loading.load_model_state_dict_by_file(
                            consumer, str(root), dtype=infinicore.bfloat16)

    def test_qualified_cpu_float_conversions_need_no_torch(self):
        # Literal bit patterns also cover ties, subnormal boundaries and
        # overflow at the LM consumer. The full arithmetic oracle lives in RT.
        f32 = [1.00048828125, 1.00146484375, 1.00390625, 1.01171875,
               2**-25, 3 * 2**-25, float("inf"), -float("inf")]
        f16_bits = [0, 0x8000, 1, 0x3c01, 0xbc00, 0x7bff, 0x7c00, 0xfc00]
        bf16_bits = [0, 0x8000, 1, 0x3f81, 0xbf80, 0x4780, 0x7f80, 0xff80]
        cases = (
            ("F32", struct.pack("<8f", *f32), infinicore.float16,
             struct.pack("<8H", 0x3c00, 0x3c02, 0x3c04, 0x3c0c, 0, 2, 0x7c00, 0xfc00)),
            ("F32", struct.pack("<8f", *f32), infinicore.bfloat16,
             struct.pack("<8H", 0x3f80, 0x3f80, 0x3f80, 0x3f82, 0x3300, 0x33c0, 0x7f80, 0xff80)),
            ("F16", struct.pack("<8H", *f16_bits), infinicore.float32,
             struct.pack("<8f", 0.0, -0.0, 2**-24, 1 + 2**-10, -1, 65504, float("inf"), -float("inf"))),
            ("F16", struct.pack("<8H", *f16_bits), infinicore.bfloat16,
             struct.pack("<8H", 0, 0x8000, 0x3380, 0x3f80, 0xbf80, 0x4780, 0x7f80, 0xff80)),
            ("BF16", struct.pack("<8H", *bf16_bits), infinicore.float32,
             struct.pack("<8f", 0.0, -0.0, 2**-133, 1 + 2**-7, -1, 65536, float("inf"), -float("inf"))),
            ("BF16", struct.pack("<8H", *bf16_bits), infinicore.float16,
             struct.pack("<8H", 0, 0x8000, 0, 0x3c08, 0xbc00, 0x7c00, 0x7c00, 0xfc00)),
        )
        for source_dtype, payload, dtype, expected in cases:
            with self.subTest(source=source_dtype, target=str(dtype)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weight.safetensors"
                bias = struct.pack("<f", 0.125)
                integer = struct.pack("<i", 7)
                _write_shard(path, [
                    ("weight", source_dtype, [2, 4], payload),
                    ("router.e_score_correction_bias", "F32", [1], bias),
                    ("indices", "I32", [1], integer),
                ])
                with patch.object(loading, "load_state_dict", side_effect=AssertionError("Torch bridge used")):
                    tensors = loading._load_checkpoint_shard(
                        str(path), dtype, (".e_score_correction_bias",), raw=True)
                gc.collect()
                converted = tensors["weight"]
                self.assertEqual(converted.dtype, dtype)
                self.assertEqual(list(converted.shape), [2, 4])
                self.assertEqual(converted.device.type, "cpu")
                self.assertIsNone(converted._owner)  # Native owning output, not a borrowed mmap.
                self.assertEqual(ctypes.string_at(converted.data_ptr(), len(expected)), expected)
                self.assertEqual(tensors["router.e_score_correction_bias"].dtype, infinicore.float32)
                self.assertEqual(ctypes.string_at(tensors["router.e_score_correction_bias"].data_ptr(), 4), bias)
                self.assertEqual(tensors["indices"].dtype, infinicore.int32)
                self.assertEqual(ctypes.string_at(tensors["indices"].data_ptr(), 4), integer)

    def test_checkpoint_cast_rejects_bad_contract_before_writing(self):
        from infinilm.lib import _infinilm

        source = infinicore.from_list([1.0, 2.0], dtype=infinicore.float32)
        bad_shape = infinicore.from_list([3.0], dtype=infinicore.float16)
        before = ctypes.string_at(bad_shape.data_ptr(), 2)
        with self.assertRaisesRegex(ValueError, "matching CPU shapes"):
            _infinilm._cast_checkpoint_tensor_(source._underlying, bad_shape._underlying)
        self.assertEqual(ctypes.string_at(bad_shape.data_ptr(), 2), before)
        same_address = infinicore.from_blob(
            source.data_ptr(), [2], dtype=infinicore.float16, device="cpu", owner=source)
        before = ctypes.string_at(source.data_ptr(), 8)
        with self.assertRaises(ValueError):
            _infinilm._cast_checkpoint_tensor_(source._underlying, same_address._underlying)
        self.assertEqual(ctypes.string_at(source.data_ptr(), 8), before)

    def test_unqualified_dtype_conversion_keeps_its_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float64.safetensors"
            _write_shard(path, [("weight", "F64", [1], struct.pack("<d", 1.25))])
            sentinel = object()
            with patch.object(loading, "_require_torch_checkpoint", return_value=object()) as required, \
                 patch.object(loading.infinicore.utils, "to_torch_dtype", return_value="bf16"), \
                 patch.object(loading, "load_state_dict", return_value=sentinel) as bridge:
                result = loading._load_checkpoint_shard(
                    str(path), infinicore.bfloat16, (), raw=True)
            self.assertIs(result, sentinel)
            required.assert_called_once_with("unqualified safetensors checkpoint fallback")
            bridge.assert_called_once()

    def test_optional_torch_missing_fails_before_fallback_or_bin_io(self):
        error = ModuleNotFoundError("No module named torch", name="torch")
        with patch("importlib.import_module", side_effect=error), \
             patch.object(loading, "load_state_dict", side_effect=AssertionError("fallback ran without Torch")):
            with self.assertRaisesRegex(ImportError, "InfiniLM\\[torch-compat\\]") as caught:
                loading._load_checkpoint_shard("unused.safetensors", infinicore.bfloat16, (), raw=False)
            self.assertIs(caught.exception.__cause__, error)
            with self.assertRaisesRegex(ImportError, "InfiniLM\\[torch-compat\\]"):
                loading._load_pytorch_bin("unused.bin")


if __name__ == "__main__":
    unittest.main()
