import importlib.util
import sys
import unittest
from pathlib import Path

from cache_test_support import MODULES, publish

SCRIPT = Path(__file__).with_name("benchmark_prefix_cache.py")
SPEC = importlib.util.spec_from_file_location("benchmark_prefix_cache", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
Scheduler = MODULES["scheduler"].Scheduler
InferenceRequest = MODULES["request"].InferenceRequest
SamplingParams = MODULES["sampling_params"].SamplingParams


class ResultAccountingTests(unittest.TestCase):
    def test_throughput_uses_total_tokens_over_shared_wall_clock(self):
        requests = [
            {"output_token_ids": list(range(32)), "ttft_seconds": 0.1},
            {"output_token_ids": list(range(32)), "ttft_seconds": 0.2},
        ]

        summary = BENCHMARK.summarize_requests(requests, wall_seconds=2.0)

        self.assertEqual(summary["generated_tokens"], 64)
        self.assertEqual(summary["throughput_tokens_per_second"], 32.0)

    def test_delivery_interval_is_the_gap_between_actual_token_times(self):
        clock = iter((10.0, 10.1, 10.15))

        result = BENCHMARK.build_request_timing(
            request_id="request-0",
            prompt_tokens=1056,
            started=10.0,
            token_ids=[41, 42],
            token_times=[next(clock), next(clock), next(clock)][1:],
        )

        self.assertAlmostEqual(result["ttft_seconds"], 0.1)
        self.assertAlmostEqual(result["delivery_intervals_seconds"][0], 0.05)
        self.assertEqual(result["output_token_ids"], [41, 42])


class SchedulerAccountingTests(unittest.TestCase):
    def make_scheduler(self, **kwargs):
        scheduler = Scheduler(block_size=256, **kwargs)
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        return scheduler

    @staticmethod
    def request(request_id, tokens, max_tokens=1):
        return InferenceRequest(
            request_id,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=max_tokens),
        )

    def test_aligned_and_one_token_tail_report_scheduler_work(self):
        for prompt_length, expected in ((1024, (768, 256)), (1025, (1024, 1))):
            with self.subTest(prompt_length=prompt_length):
                scheduler = self.make_scheduler(num_blocks=16)
                publish(scheduler.cache_manager, [11] * 1024)
                request = self.request("reuse", [11] * prompt_length)
                scheduler.add_request(request)
                accounting = BENCHMARK.SchedulerAccounting(scheduler)
                accounting.install()

                scheduler.schedule()
                accounting.require_exactly_once([request.request_id])

                self.assertEqual(
                    accounting.by_request[request.request_id],
                    {
                        "local_cached_tokens": expected[0],
                        "prefill_tokens": expected[1],
                    },
                )

    def test_no_reuse_reports_full_prefill(self):
        scheduler = self.make_scheduler(num_blocks=16)
        publish(scheduler.cache_manager, [11] * 1024)
        request = self.request("new", [22] * 1025)
        scheduler.add_request(request)
        accounting = BENCHMARK.SchedulerAccounting(scheduler)
        accounting.install()

        scheduler.schedule()
        accounting.require_exactly_once([request.request_id])

        self.assertEqual(
            accounting.by_request[request.request_id],
            {"local_cached_tokens": 0, "prefill_tokens": 1025},
        )

    def test_budget_rejected_probe_is_excluded(self):
        scheduler = self.make_scheduler(
            num_blocks=16, max_batch_size=2, max_num_batched_tokens=1024
        )
        publish(scheduler.cache_manager, [11] * 1024)
        first = self.request("first", [11] * 1025)
        rejected = self.request("budget-rejected", [11] * 1024 + [22] * 1024)
        scheduler.add_request(first)
        scheduler.add_request(rejected)
        accounting = BENCHMARK.SchedulerAccounting(scheduler)
        accounting.install()

        scheduler.schedule()

        self.assertEqual(set(accounting.by_request), {"first"})
        self.assertNotIn(rejected.request_id, accounting.by_request)

    def test_capacity_rejected_probe_is_excluded(self):
        scheduler = self.make_scheduler(num_blocks=2)
        publish(scheduler.cache_manager, [11] * 256)
        rejected = self.request("capacity-rejected", [11] * 256 + [22], max_tokens=512)
        scheduler.add_request(rejected)
        accounting = BENCHMARK.SchedulerAccounting(scheduler)
        accounting.install()

        self.assertIsNone(scheduler.schedule())

        self.assertEqual(accounting.by_request, {})

    def test_missing_or_duplicate_admission_accounting_fails(self):
        scheduler = self.make_scheduler(num_blocks=16)
        request = self.request("duplicate", [11] * 257)
        scheduler.add_request(request)
        accounting = BENCHMARK.SchedulerAccounting(scheduler)
        accounting.install()
        scheduler.schedule()

        with self.assertRaisesRegex(RuntimeError, "missing"):
            accounting.require_exactly_once(["unknown"])
        accounting.admission_counts[request.request_id] = 2
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            accounting.require_exactly_once([request.request_id])


class RepeatArtifactTests(unittest.TestCase):
    def test_mixed_repeat_outcomes_preserve_every_payload(self):
        records = [
            {"repeat": 0, "status": "success", "payload": {"trace_sha256": "abc"}},
            {
                "repeat": 1,
                "status": "timeout",
                "timeout_seconds": 15,
                "stdout": "partial output",
                "stderr": "",
            },
        ]

        result = BENCHMARK.assemble_repeat_artifact(
            records, {"scenario": "hot-cold", "repeat_timeout_seconds": 15}
        )

        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["repeats"], records)
        self.assertEqual(result["successful_repeats"], 1)
        self.assertEqual(result["failed_repeats"], 1)

    def test_child_deadline_reports_timeout(self):
        returncode, stdout, stderr, timed_out = BENCHMARK._run_child(
            [sys.executable, "-c", "import time; time.sleep(10)"], 0.05
        )

        self.assertTrue(timed_out)
        self.assertIsNone(returncode)
        self.assertEqual((stdout, stderr), ("", ""))


class TraceTests(unittest.TestCase):
    def test_no_reuse_requests_have_distinct_first_blocks(self):
        token_sources = [list(range(10, 410)), list(range(510, 910))]

        trace = BENCHMARK.build_trace(
            "no-reuse", token_sources, seed=7, request_count=128
        )

        first_blocks = {tuple(item["prompt_token_ids"][:256]) for item in trace}
        self.assertEqual(len(first_blocks), 128)
        self.assertTrue(all(len(item["prompt_token_ids"]) == 1056 for item in trace))

    def test_shared_requests_reuse_one_prefix_but_keep_unique_tails(self):
        token_sources = [list(range(10, 410)), list(range(510, 910))]

        trace = BENCHMARK.build_trace("shared", token_sources, seed=7, request_count=8)

        self.assertEqual(
            len({tuple(item["prompt_token_ids"][:1024]) for item in trace}), 1
        )
        self.assertEqual(
            len({tuple(item["prompt_token_ids"][1024:]) for item in trace}), 8
        )

    def test_seed_is_repeatable_and_changes_every_scenario_trace(self):
        token_sources = [list(range(10, 410)), list(range(510, 910))]
        scenarios = (
            "hot-cold",
            "hot-shift",
            "no-reuse",
            "over-capacity",
            "mixed-length",
            "shared",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                first = BENCHMARK.build_trace(scenario, token_sources, seed=7)
                repeated = BENCHMARK.build_trace(scenario, token_sources, seed=7)
                changed = BENCHMARK.build_trace(scenario, token_sources, seed=8)
                self.assertEqual(first, repeated)
                self.assertNotEqual(
                    [item["prompt_token_ids"] for item in first],
                    [item["prompt_token_ids"] for item in changed],
                )
                self.assertEqual(
                    [
                        (item["phase"], item["group"], len(item["prompt_token_ids"]))
                        for item in first
                    ],
                    [
                        (item["phase"], item["group"], len(item["prompt_token_ids"]))
                        for item in changed
                    ],
                )


if __name__ == "__main__":
    unittest.main()
