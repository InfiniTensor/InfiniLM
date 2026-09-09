import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SpeculativeRunnerContractsTest(unittest.TestCase):
    def test_primary_target_only_requests_all_positions_during_prefill(self) -> None:
        source = (
            ROOT / "python/infinilm/llm/model_runner/speculative_runner.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SpeculativeRunner"
        )
        method = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )

        assignment = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "target_model_input"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "sample_all_positions"
                for target in node.targets
            )
        )
        self.assertIsInstance(assignment.value, ast.Attribute)
        self.assertEqual(assignment.value.attr, "is_prefill")
        self.assertIsInstance(assignment.value.value, ast.Name)
        self.assertEqual(assignment.value.value.id, "scheduler_output")

        target_call = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward_raw"
            and any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "target_model_input"
                for keyword in node.keywords
            )
        )
        self.assertIsInstance(target_call.func.value, ast.Attribute)
        self.assertEqual(target_call.func.value.attr, "target_model_engine")

    def test_single_token_verification_uses_last_position_graph_path(self) -> None:
        source = (
            ROOT / "python/infinilm/llm/model_runner/speculative_runner.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SpeculativeRunner"
        )
        method = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_paged_verify_batch_input"
        )
        assignment = next(
            node
            for node in method.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "sample_all_positions"
                for target in node.targets
            )
        )
        expression = compile(
            ast.Expression(assignment.value),
            str(ROOT / "speculative_runner.py"),
            "eval",
        )

        def requires_all_positions(lengths: list[int]) -> bool:
            namespace = {
                "any": any,
                "len": len,
                "candidates": [{"draft_tokens": [0] * length} for length in lengths],
            }
            return bool(eval(expression, namespace))

        self.assertFalse(requires_all_positions([1]))
        self.assertFalse(requires_all_positions([1, 1]))
        self.assertTrue(requires_all_positions([2]))
        self.assertTrue(requires_all_positions([1, 2]))

        returned = next(
            node for node in reversed(method.body) if isinstance(node, ast.Return)
        )
        self.assertIsInstance(returned.value, ast.Dict)
        entries = {
            key.value: value
            for key, value in zip(returned.value.keys, returned.value.values)
            if isinstance(key, ast.Constant)
        }
        self.assertIn("sample_all_positions", entries)
        self.assertIsInstance(entries["sample_all_positions"], ast.Name)
        self.assertEqual(entries["sample_all_positions"].id, "sample_all_positions")

    def test_draft_forward_disables_all_position_sampling(self) -> None:
        source = (
            ROOT / "python/infinilm/llm/model_runner/speculative_runner.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SpeculativeRunner"
        )
        method = next(
            node
            for node in runner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_draft_eagle_tokens_batch"
        )
        draft_call = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward_raw"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "draft_model_engine"
        )
        keywords = {keyword.arg: keyword.value for keyword in draft_call.keywords}
        sample_all_positions = keywords["sample_all_positions"]
        self.assertIsInstance(sample_all_positions, ast.Constant)
        self.assertIs(sample_all_positions.value, False)


if __name__ == "__main__":
    unittest.main()
