from types import SimpleNamespace
import unittest

from infinilm.llm.model_runner.model_runner import (
    split_pipeline_microbatches,
    uniform_sampling_params,
)
from infinilm.llm.scheduler import SchedulerOutput


def request(request_id, temperature=1.0, top_p=0.8, top_k=1):
    return SimpleNamespace(
        request_id=request_id,
        sampling_params=SimpleNamespace(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
    )


def request_ids(outputs):
    return [
        [request.request_id for request in output.scheduled_requests]
        for output in outputs
    ]


class PipelineMicrobatchTest(unittest.TestCase):
    def test_split_preserves_request_order_and_phase(self):
        scheduler_output = SchedulerOutput(
            [request(str(index)) for index in range(5)],
            is_prefill=True,
        )

        outputs = split_pipeline_microbatches(scheduler_output, max_requests=2)

        self.assertEqual(request_ids(outputs), [["0", "1"], ["2", "3"], ["4"]])
        self.assertTrue(all(output.is_prefill for output in outputs))

    def test_split_separates_sampling_parameters(self):
        scheduler_output = SchedulerOutput(
            [
                request("a", top_k=1),
                request("b", top_k=1),
                request("c", top_k=5),
                request("d", top_k=5),
                request("e", temperature=0.5, top_k=5),
            ]
        )

        outputs = split_pipeline_microbatches(scheduler_output, max_requests=8)

        self.assertEqual(request_ids(outputs), [["a", "b"], ["c", "d"], ["e"]])
        for output in outputs:
            keys = {
                (
                    req.sampling_params.temperature,
                    req.sampling_params.top_p,
                    req.sampling_params.top_k,
                )
                for req in output.scheduled_requests
            }
            self.assertEqual(len(keys), 1)

    def test_split_preserves_connector_metadata(self):
        scheduler_output = SchedulerOutput([request("a"), request("b")])
        marker = object()
        scheduler_output.kv_connector_metadata = marker

        outputs = split_pipeline_microbatches(scheduler_output, max_requests=1)
        self.assertTrue(
            all(output.kv_connector_metadata is marker for output in outputs)
        )

    def test_split_rejects_invalid_size(self):
        scheduler_output = SchedulerOutput([request("a")])
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            split_pipeline_microbatches(scheduler_output, max_requests=0)

    def test_uniform_sampling_params(self):
        scheduler_output = SchedulerOutput(
            [request("a", top_k=5), request("b", top_k=5)]
        )
        params = uniform_sampling_params(scheduler_output)
        self.assertIsNotNone(params)
        self.assertEqual(params.top_k, 5)

    def test_mixed_sampling_params_fall_back(self):
        scheduler_output = SchedulerOutput(
            [request("a", top_k=1), request("b", top_k=5)]
        )
        self.assertIsNone(uniform_sampling_params(scheduler_output))


if __name__ == "__main__":
    unittest.main()
