"""Progress under sustained arrivals and cancellation, without model loading."""

import unittest

from infinilm.llm.request import RequestStatus
from test_chunked_prefill_sched import make_request, make_scheduler


def complete(scheduler, output, finish=False):
    for req in output.scheduled_requests:
        if output.is_prefill:
            req.num_computed_tokens = req.get_prompt_length()
        req.append_generated_token_id(42)
        if finish:
            req.status = RequestStatus.FINISHED
    scheduler.complete_requests(output.scheduled_requests)


class SchedulerFairnessTest(unittest.TestCase):
    def test_continuous_arrivals_allow_decode_without_chunking(self):
        s = make_scheduler(max_batch_size=1, num_blocks=1024)
        old = make_request('old', 8, max_tokens=128)
        s.add_request(old)
        complete(s, s.schedule())
        phases = []
        for i in range(40):
            s.add_request(make_request('new-'+str(i), 8))
            out = s.schedule()
            self.assertIsNotNone(out)
            phases.append(out.is_prefill)
            complete(s, out, finish=out.is_prefill)
        self.assertEqual(phases, [False, True] * 20)
        self.assertEqual(old.get_num_generated_tokens(), 21)

    def test_canceled_decode_falls_back_to_waiting_in_same_call(self):
        s = make_scheduler(max_batch_size=1)
        old = make_request('old', 8)
        s.add_request(old)
        complete(s, s.schedule())
        old.status = RequestStatus.CANCELED
        s.add_request(make_request('new', 8))
        out = s.schedule()
        self.assertIsNotNone(out)
        self.assertTrue(out.is_prefill)
        self.assertEqual(out.scheduled_requests[0].request_id, 'new')

    def test_running_reservation_scanned_once_per_admission_batch(self):
        s = make_scheduler(max_batch_size=4, num_blocks=1024)
        s.add_request(make_request('old', 8))
        complete(s, s.schedule())
        complete(s, s.schedule())
        calls = []
        original = s._get_running_decode_blocks
        def tracked():
            calls.append(True)
            return original()
        s._get_running_decode_blocks = tracked
        for i in range(4):
            s.add_request(make_request('new-'+str(i), 8))
        out = s.schedule()
        self.assertTrue(out.is_prefill)
        self.assertEqual(len(out.scheduled_requests), 4)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
