import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ModelingUtilsContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (ROOT / "python/infinilm/modeling_utils.py").read_text(
            encoding="utf-8"
        )
        cls.module = ast.parse(source)

    def _function(self, name: str) -> ast.FunctionDef:
        return next(
            node
            for node in self.module.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_zip_pytorch_checkpoints_use_memory_mapping(self) -> None:
        loader = self._function("_load_pytorch_bin")
        zip_guard = next(node for node in loader.body if isinstance(node, ast.If))
        torch_load = next(
            node
            for node in ast.walk(loader)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        )

        self.assertIn("zipfile.is_zipfile(file_path)", ast.unparse(zip_guard.test))
        mmap_assignment = next(
            node
            for node in zip_guard.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "load_kwargs"
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "mmap"
        )
        self.assertIsInstance(mmap_assignment.value, ast.Constant)
        self.assertIs(mmap_assignment.value.value, True)
        self.assertTrue(
            any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "load_kwargs"
                for keyword in torch_load.keywords
            )
        )

    def test_pytorch_bin_entry_points_share_the_mmap_loader(self) -> None:
        for function_name in (
            "load_model_state_dict_by_file",
            "load_model_state_dict_by_tensor",
        ):
            with self.subTest(function=function_name):
                function = self._function(function_name)
                calls = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_load_pytorch_bin"
                ]
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
