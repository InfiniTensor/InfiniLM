"""Run the real generation loop with CPU fixtures, without native libraries."""

import ast
import contextlib
import io
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "python/infinilm/infer_engine.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
CONFIG = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "GenerationConfig"
)
GENERATE = next(
    method
    for cls in TREE.body
    if isinstance(cls, ast.ClassDef)
    for method in cls.body
    if isinstance(method, ast.FunctionDef) and method.name == "generate"
)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.reads = 0

    def view(self, shape):
        return Tensor(self.values.reshape(shape))

    def to_numpy(self):
        self.reads += 1
        return self.values.copy()


class PagedMetadata:
    def __init__(self, batch, capacity):
        self.batch = batch

    def update(self, past):
        return (Tensor([past] * self.batch),) * 6


NAMESPACE = {
    "dataclass": dataclass,
    "time": time,
    "infinicore": SimpleNamespace(
        from_list=lambda values, **kwargs: Tensor(values), int32="int32", int64="int64"
    ),
    "_PagedDecodeMetadataBuffers": PagedMetadata,
}
exec(
    compile(ast.Module(body=[CONFIG, GENERATE], type_ignores=[]), str(SOURCE), "exec"),
    NAMESPACE,
)
GenerationConfig = NAMESPACE["GenerationConfig"]


class Engine:
    generate = NAMESPACE["generate"]
    position_id_axes = 1
    has_mamba_cache = False

    def __init__(self, steps, *, eos=(2,), paged=False):
        self.steps = [Tensor(s) for s in steps]
        self.eos_token_id = list(eos)
        self.enable_paged_attn = paged
        self.calls = []

    def __call__(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        return self.steps[index]

    def get_cache_config(self):
        return SimpleNamespace(block_size=lambda: 16)


class StaticBatchEosTest(unittest.TestCase):
    def run_case(
        self, steps, *, eos=(2,), paged=False, enabled=True, stop=True, **kwargs
    ):
        engine = Engine(steps, eos=eos, paged=paged)
        config = GenerationConfig(
            max_new_tokens=len(steps), stop_on_eos=stop, stop_batch_on_eos=enabled
        )
        output = engine.generate(Tensor([[10, 11]] * len(steps[0])), config, **kwargs)
        return engine, [x.values.tolist() for x in output]

    def test_disabled_by_default(self):
        self.assertFalse(GenerationConfig().stop_batch_on_eos)

    def test_staggered_completion_is_sticky(self):
        steps = [[2, 7], [8, 2], [9, 9]]
        engine, output = self.run_case(steps)
        self.assertEqual(output, steps[:2])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[1]["input_ids"].values.tolist(), [[2], [7]])

    def test_one_unfinished_row_keeps_full_budget(self):
        steps = [[2, 7], [8, 8], [9, 9]]
        engine, output = self.run_case(steps)
        self.assertEqual(output, steps)
        self.assertEqual(len(engine.calls), 3)

    def test_stop_on_eos_false_keeps_fixed_workload(self):
        steps = [[2, 2], [8, 8], [9, 9]]
        engine, output = self.run_case(steps, stop=False)
        self.assertEqual(output, steps)
        self.assertEqual(sum(t.reads for t in engine.steps), 0)

    def test_disabled_option_does_not_read_tokens(self):
        steps = [[2, 2], [8, 8]]
        engine, output = self.run_case(steps, enabled=False)
        self.assertEqual(output, steps)
        self.assertEqual(sum(t.reads for t in engine.steps), 0)

    def test_multiple_eos_values(self):
        _, output = self.run_case([[2, 3], [9, 9]], eos=(2, 3))
        self.assertEqual(output, [[2, 3]])

    def test_no_eos_configuration_keeps_budget(self):
        engine, output = self.run_case([[2, 2], [9, 9]], eos=())
        self.assertEqual(len(output), 2)
        self.assertEqual(sum(t.reads for t in engine.steps), 0)

    def test_single_sequence_behavior_is_unchanged(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                _, output = self.run_case([[7], [2], [9]], enabled=enabled)
                self.assertEqual(output, [[7], [2]])

    def test_paged_path_is_unchanged(self):
        engine, output = self.run_case([[2, 2], [9, 9]], paged=True)
        self.assertEqual(len(output), 2)
        self.assertEqual(sum(t.reads for t in engine.steps), 0)

    def test_multimodal_path_is_unchanged(self):
        engine, output = self.run_case([[2, 2], [9, 9]], pixel_values=object())
        self.assertEqual(len(output), 2)
        self.assertEqual(sum(t.reads for t in engine.steps), 0)

    def test_batch_sixteen_stops_at_last_row(self):
        steps = [[2] * 15 + [7], [9] * 15 + [2], [9] * 16]
        _, output = self.run_case(steps)
        self.assertEqual(output, steps[:2])

    def test_changed_output_cardinality_is_rejected(self):
        engine = Engine([[2]])
        config = GenerationConfig(max_new_tokens=1, stop_batch_on_eos=True)
        with self.assertRaisesRegex(ValueError, "cardinality"):
            engine.generate(Tensor([[10], [11]]), config)

    def test_first_step_completion_is_included_in_timing(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _, output = self.run_case([[2, 2], [9, 9]], _measure_and_log_time=True)
        self.assertEqual(output, [[2, 2]])
        self.assertIn("Per_Batch_New_Tokens=1", buffer.getvalue())

    def test_final_decode_step_is_included_in_timing(self):
        buffer = io.StringIO()
        with (
            contextlib.redirect_stdout(buffer),
            patch.object(time, "perf_counter", side_effect=[0.0, 0.01, 0.01, 0.03]),
        ):
            _, output = self.run_case(
                [[2, 7], [8, 2], [9, 9]], _measure_and_log_time=True
            )
        self.assertEqual(output, [[2, 7], [8, 2]])
        self.assertIn("Per_Batch_New_Tokens=2", buffer.getvalue())
        self.assertIn("Generation completed in 30.0 ms", buffer.getvalue())
        self.assertIn("Decode  Avg ITL: 20.0 ms", buffer.getvalue())

    def test_effective_sequences_match_disabled_reference(self):
        steps = [[7, 8], [2, 9], [10, 2], [11, 12]]
        _, baseline = self.run_case(steps, enabled=False)
        _, candidate = self.run_case(steps)
        for row in range(2):
            reference = [step[row] for step in baseline]
            expected = reference[: reference.index(2) + 1]
            actual = [step[row] for step in candidate]
            self.assertEqual(actual[: actual.index(2) + 1], expected)


if __name__ == "__main__":
    unittest.main()
