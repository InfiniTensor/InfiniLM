"""Test the pure routing helper without requiring a compiled GPU extension."""

import ast
import unittest
from pathlib import Path

source = (
    Path(__file__).resolve().parents[3] / "python/infinilm/processors/lfm2_processor.py"
)
tree = ast.parse(source.read_text(encoding="utf-8"))
function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "static_short_conv_state_indices"
)
namespace = {}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
    namespace,
)
route = namespace["static_short_conv_state_indices"]


class Lfm2StateRoutingTest(unittest.TestCase):
    def test_new_requests_never_read_previous_terminal_state(self):
        # A prefill, A decode, B prefill, B decode, A prefill.
        self.assertEqual(
            [route(stage, 0, 1) for stage in (True, False, True, False, True)],
            [([0], [1]), ([1], [1]), ([0], [1]), ([1], [1]), ([0], [1])],
        )

    def test_unimplemented_prefix_and_batch_modes_are_rejected(self):
        for arguments in ((True, 5, 1), (True, 0, 0), (True, 0, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                route(*arguments)


if __name__ == "__main__":
    unittest.main()
