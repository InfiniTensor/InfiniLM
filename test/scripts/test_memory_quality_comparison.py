"""CPU regressions for invalid and divergent memory-quality captures."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "memory_quality_comparison",
    Path(__file__).resolve().parents[1] / "bench/compare_qwen3_memory_quality.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MemoryQualityComparisonTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = [
            Path(self.directory.name) / name
            for name in ("control.json", "optimized.json")
        ]
        case = {
            "dataset": "qasper",
            "index": 0,
            "id": "document",
            "token_count": 10001,
            "answers": ["answer"],
            "token_ids": [1, 2],
        }
        self.records = [
            {
                "status": "COMPLETE",
                "framework": "infinilm",
                "control": flag,
                "cases_sha256": "a" * 64,
                "model": "local-model",
                "cases": [copy.deepcopy(case)],
            }
            for flag in (True, False)
        ]
        self.arrays = [{"0": np.array([1.0, 2.0, 3.0])} for _ in range(2)]

    def run_comparison(self, strict=False):
        for path, record, arrays in zip(self.paths, self.records, self.arrays):
            path.write_text(json.dumps(record), encoding="utf-8")
            np.savez(path.with_suffix(".npz"), **arrays)
        return MODULE.compare(*self.paths, require_identical_tokens=strict)

    def test_identical_capture_passes_strict(self):
        report = self.run_comparison(strict=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["cases"][0]["relative_l2"], 0)

    def test_invalid_cli_replaces_stale_pass_and_does_not_overwrite_input(self):
        self.run_comparison()
        self.paths[1].write_text("{}", encoding="utf-8")
        result = Path(self.directory.name) / "report.json"
        result.write_text('{"status": "PASS"}', encoding="utf-8")
        command = [
            sys.executable,
            "-B",
            str(SPEC.origin),
            "--control",
            str(self.paths[0]),
            "--optimized",
            str(self.paths[1]),
            "--result",
        ]
        process = subprocess.run(command + [str(result)], capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(
            json.loads(result.read_text(encoding="utf-8"))["status"], "INVALID"
        )
        original = self.paths[0].read_bytes()
        process = subprocess.run(command + [str(self.paths[0])], capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(self.paths[0].read_bytes(), original)

    def test_numerical_regression_fails(self):
        self.arrays[1]["0"] *= -1
        self.assertEqual(self.run_comparison()["status"], "FAIL")

    def test_corrupted_npz_and_overflow_norms_are_invalid(self):
        result = Path(self.directory.name) / "report.json"
        command = [
            sys.executable,
            "-B",
            str(SPEC.origin),
            "--control",
            str(self.paths[0]),
            "--optimized",
            str(self.paths[1]),
            "--result",
            str(result),
        ]
        for corrupt in (False, True):
            self.arrays[1]["0"] = np.full(3, 1e308) if not corrupt else np.ones(3)
            for path, record, arrays in zip(self.paths, self.records, self.arrays):
                path.write_text(json.dumps(record), encoding="utf-8")
                np.savez(path.with_suffix(".npz"), **arrays)
            if corrupt:
                self.paths[1].with_suffix(".npz").write_bytes(b"PK\x03\x04broken")
            process = subprocess.run(command, capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(
                json.loads(result.read_text(encoding="utf-8"))["status"], "INVALID"
            )

    def test_greedy_drift_is_reported_and_can_be_required(self):
        self.records[1]["cases"][0]["token_ids"] = [2, 1]
        self.assertFalse(self.run_comparison()["cases"][0]["generated_ids_equal"])
        self.assertEqual(self.run_comparison(strict=True)["status"], "FAIL")

    def test_incomplete_empty_and_swapped_captures_rejected(self):
        for key, value in (("status", "RUNNING"), ("cases", []), ("control", True)):
            with self.subTest(key=key):
                previous = self.records[1][key]
                self.records[1][key] = value
                with self.assertRaises(ValueError):
                    self.run_comparison()
                self.records[1][key] = previous

    def test_different_protocol_model_and_case_rejected(self):
        for key, value in (("cases_sha256", "b" * 64), ("model", "other-model")):
            with self.subTest(key=key):
                previous = self.records[1][key]
                self.records[1][key] = value
                with self.assertRaises(ValueError):
                    self.run_comparison()
                self.records[1][key] = previous
        self.records[1]["cases"][0]["id"] = "other-document"
        with self.assertRaises(ValueError):
            self.run_comparison()

    def test_missing_and_extra_logit_keys_rejected(self):
        for arrays in ({}, {"0": np.ones(3), "1": np.ones(3)}):
            self.arrays[1] = arrays
            with self.assertRaises(ValueError):
                self.run_comparison()

    def test_invalid_logit_vectors_rejected(self):
        for array in (
            np.array([]),
            np.ones((1, 3)),
            np.ones(4),
            np.array([1, 2, 3]),
            np.array([1.0, np.nan, 3.0]),
            np.array([1.0, np.inf, 3.0]),
        ):
            with self.subTest(array=array):
                self.arrays[1]["0"] = array
                with self.assertRaises(ValueError):
                    self.run_comparison()

    def test_empty_and_out_of_vocabulary_generation_rejected(self):
        for tokens in ([], [3], [-1], [True]):
            self.records[1]["cases"][0]["token_ids"] = tokens
            with self.assertRaises(ValueError):
                self.run_comparison()


if __name__ == "__main__":
    unittest.main()
