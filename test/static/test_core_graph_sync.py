import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    signature_start = source.index(signature)
    body_start = source.index("{", signature_start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start : index + 1]
    raise AssertionError(f"Unterminated function body for {signature}")


class CoreGraphSyncTest(unittest.TestCase):
    def test_graph_replay_is_partitioned_by_capture_safety(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        source = read_source("csrc/infinicore/src/graph/graph.cc")
        instantiate = function_body(source, "void Graph::instantiate()")
        run = function_body(source, "void Graph::run() const")

        self.assertIn(
            "virtual bool is_device_graph_capture_safe() const { return true; }",
            header,
        )
        for contract in (
            "bool capture_safe;",
            "std::vector<std::shared_ptr<GraphOperator>> ops;",
            "std::unique_ptr<DeviceGraph> device_graph_;",
            "std::vector<std::unique_ptr<Segment>> segments_;",
        ):
            self.assertIn(contract, header)

        warmup = instantiate.index("for (size_t iter = 0; iter < 5; ++iter)")
        disable = instantiate.index("INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS")
        partition = instantiate.index("op->is_device_graph_capture_safe()")
        self.assertLess(warmup, disable)
        self.assertLess(disable, partition)
        self.assertIn("segments_.back()->ops.push_back(op)", instantiate)
        self.assertIn("if (!segment->capture_safe)", instantiate)
        self.assertIn("segment->run()", instantiate)
        self.assertIn("StreamCaptureGuard", instantiate)
        self.assertIn("rt_runtime::GraphInstantiate", instantiate)
        self.assertIn("INFINICORE_GRAPH_DEBUG", instantiate)
        self.assertIn("host_segments", instantiate)

        self.assertIn("if (segments_.empty())", run)
        self.assertIn("segment->run()", run)

    def test_segment_cleanup_keeps_runtime_and_allocation_leases_alive(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        source = read_source("csrc/infinicore/src/graph/graph.cc")
        destructor = function_body(source, "Graph::~Graph() noexcept")

        runtime_lease = header.index("runtime_lease_")
        allocation_lease = header.index("allocation_lease_")
        operators = header.index("op_list_")
        segments = header.index("segments_")
        self.assertLess(runtime_lease, operators)
        self.assertLess(allocation_lease, operators)
        self.assertLess(runtime_lease, segments)
        self.assertLess(allocation_lease, segments)
        self.assertIn("runtime_lease_->syncStreamForCleanup()", destructor)
        self.assertIn("Graph::Segment::~Segment() noexcept = default", source)


if __name__ == "__main__":
    unittest.main()
