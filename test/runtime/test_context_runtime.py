import threading
import unittest

import infinicore
import torch


class RuntimeStreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_count = infinicore.get_device_count("cuda")
        if cls.device_count == 0:
            raise unittest.SkipTest("NVIDIA device is required")

    def tearDown(self):
        infinicore.set_device("cuda:0")

    def test_stream_is_stable_on_one_device(self):
        infinicore.set_device("cuda:0")

        first = infinicore.get_stream()
        second = infinicore.get_stream()
        infinicore.zeros((4,), device="cuda:0")
        infinicore.sync_stream()

        self.assertNotEqual(first, 0)
        self.assertEqual(first, second)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_switching_devices_reuses_each_runtime_stream(self):
        if self.device_count < 2:
            self.skipTest("two NVIDIA devices are required")

        infinicore.set_device("cuda:0")
        stream_0 = infinicore.get_stream()
        keep_runtime_0 = infinicore.zeros((4,), device="cuda:0")
        infinicore.sync_stream()

        infinicore.set_device("cuda:1")
        stream_1 = infinicore.get_stream()
        keep_runtime_1 = infinicore.zeros((4,), device="cuda:1")
        infinicore.sync_stream()

        infinicore.set_device("cuda:0")
        self.assertEqual(infinicore.get_stream(), stream_0)
        self.assertNotEqual(stream_0, stream_1)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertIsNotNone(keep_runtime_0)
        self.assertIsNotNone(keep_runtime_1)

    def test_threads_own_distinct_streams_on_one_device(self):
        results = self._collect_thread_streams((0, 0))

        self.assertNotEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][1], 0)
        self.assertEqual(results[1][1], 0)

    def test_threads_select_devices_independently(self):
        if self.device_count < 2:
            self.skipTest("two NVIDIA devices are required")

        infinicore.set_device("cuda:0")
        results = self._collect_thread_streams((0, 1))

        self.assertEqual(results[0][1], 0)
        self.assertEqual(results[1][1], 1)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_explicit_set_device_restores_external_cuda_switch(self):
        if self.device_count < 2:
            self.skipTest("two NVIDIA devices are required")

        infinicore.set_device("cuda:0")
        torch.cuda.set_device(1)
        self.assertEqual(torch.cuda.current_device(), 1)

        infinicore.set_device("cuda:0")
        self.assertEqual(torch.cuda.current_device(), 0)

    def _collect_thread_streams(self, device_indices):
        barrier = threading.Barrier(len(device_indices))
        results = [None] * len(device_indices)
        errors = []

        def worker(slot, device_index):
            try:
                target = f"cuda:{device_index}"
                infinicore.set_device(target)
                first = infinicore.get_stream()
                keep_runtime = infinicore.zeros((4,), device=target)
                infinicore.sync_stream()
                barrier.wait(timeout=30)
                second = infinicore.get_stream()
                results[slot] = (second, torch.cuda.current_device())
                self.assertEqual(first, second)
                self.assertIsNotNone(keep_runtime)
            except BaseException as error:
                errors.append(error)
                barrier.abort()

        threads = [
            threading.Thread(target=worker, args=(slot, device_index))
            for slot, device_index in enumerate(device_indices)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=40)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        if errors:
            raise errors[0]
        return results


if __name__ == "__main__":
    unittest.main()
