"""Unexecuted CPU contract tests for bounded-hash admission and feedback.

The costs below are synthetic arithmetic fixtures, not a calibration profile.
These tests were added during adaptation and have status NOT_RUN.
"""

import asyncio
import importlib
import importlib.util
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "python/infinilm/routing/bounded_hash"
)


def load_router_package():
    """Load the actual pure-Python package without the native model package."""
    name = "bounded_hash_router_under_test"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    package = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: package}):
        spec.loader.exec_module(package)
        modules = {
            module: importlib.import_module(name + "." + module)
            for module in (
                "cache_router_v3", "cache_router", "affinity_router", "stable_router"
            )
        }
    return modules


MODULES = load_router_package()
Router = MODULES["cache_router_v3"].Router


def item(name, output=24):
    """Describe a page-aligned token contract from a hypothetical backend."""
    return {
        "id": name, "key": "shared-document", "prompt_tokens": 1000,
        "output_tokens": output, "shared_tokens": 768,
        "prefix_digest": "verified-token-prefix",
    }


def feedback(instance, epoch, produced=1, age_ms=0):
    """Represent CPU metadata published by the backend's owner thread."""
    return {
        "choices": [{"delta": {"content": "token"}}],
        "x_infinilm": {
            "backend_instance_id": instance,
            "generated_tokens": produced,
            "cache_snapshot": {
                "backend_instance_id": instance, "epoch": epoch,
                "prefixes": {"verified-token-prefix": 768}, "age_ms": age_ms,
            },
        },
    }


@contextmanager
def clock(now):
    """Control routing estimates without changing the asyncio loop's clock."""
    value = [now]
    routing_time = SimpleNamespace(monotonic=lambda: value[0])
    with ExitStack() as stack:
        for name in ("cache_router", "affinity_router", "stable_router"):
            stack.enter_context(patch.object(MODULES[name], "time", routing_time))
        yield value


class BoundedHashRouterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.routers = []

    def router(self, count=1, prefill_slope=10, **options):
        """Construct the real router with deliberately synthetic cost data."""
        model = {
            "schema_version": 1,
            "prefill_coef": [1, prefill_slope, 0],
            "decode_step_ms": {"1": 1, "2": 1, "3": 1},
            "decode_reference_context_tokens": 1000,
        }
        defaults = {
            "page_size": 256, "kv_tokens": 4096, "max_inflight": 3,
            "cache_ttl_s": 5, "queue_timeout_s": 5,
        }
        defaults.update(options)
        router = Router(
            [f"http://replica-{i}.invalid" for i in range(count)],
            phase_model=model, **defaults,
        )
        for worker in range(count):
            router.observe_backend(worker, f"instance-{worker}")
        self.routers.append(router)
        return router

    def tearDown(self):
        for router in self.routers:
            router.close()
            for lease in list(router.active.values()):
                router.release(lease, False)

    async def test_hints_never_reduce_full_kv_or_inflight_reservations(self):
        for max_inflight, capacity, slots in ((3, 2048, 2), (1, 4096, 1)):
            with self.subTest(max_inflight=max_inflight, capacity=capacity):
                router = self.router(max_inflight=max_inflight, kv_tokens=capacity)
                active = []
                for index in range(slots):
                    lease = await router.acquire(item(f"active-{index}"))
                    active.append(lease)
                    self.assertEqual(lease.reserved_tokens, 1024)
                    router.progress(lease, feedback("instance-0", index + 1))
                waiting = asyncio.create_task(router.acquire(item("blocked")))
                await asyncio.sleep(0)
                self.assertFalse(waiting.done())
                self.assertEqual(router.snapshot()["active"], [slots])
                self.assertEqual(router.snapshot()["kv_reserved_tokens"], [1024 * slots])
                router.release(active.pop(0), True)
                admitted = await asyncio.wait_for(waiting, 1)
                active.append(admitted)
                self.assertEqual(admitted.reserved_tokens, 1024)
                self.assertLessEqual(router.reserved[0], capacity)
                self.assertLessEqual(router.workers[0].active_load, max_inflight)
                for lease in active:
                    router.release(lease, True)

    async def test_a_hot_prefix_keeps_at_most_two_persistent_owners(self):
        router = self.router(count=3, max_inflight=1)
        for index in range(16):
            lease = await router.acquire(item(f"learn-{index}"))
            router.progress(lease, feedback(f"instance-{lease.worker}", index + 1))
            router.release(lease, True)
        first = await router.acquire(item("owner"))
        router.progress(first, feedback(f"instance-{first.worker}", 17))
        second = await router.acquire(item("secondary"))
        router.progress(second, feedback(f"instance-{second.worker}", 18))
        third = await router.acquire(item("temporary-spill"))
        self.assertEqual(len({first.worker, second.worker, third.worker}), 3)
        self.assertEqual(len(third.route["stable_owners"]), 2)
        self.assertNotIn(third.worker, third.route["stable_owners"])
        self.assertEqual(router.snapshot()["persistent_replicas"], 2)
        for lease in (first, second, third):
            router.release(lease, True)

    async def test_snapshot_age_and_same_epoch_cannot_refresh_ttl(self):
        router = self.router()
        with clock(1000) as now:
            lease = await router.acquire(item("publishing"))
            router.progress(lease, feedback("instance-0", 1, age_ms=4900))
            now[0] = 1000.2
            router.progress(lease, feedback("instance-0", 1, produced=2, age_ms=0))
            expired = await router.acquire(item("same-epoch-expired"))
            self.assertEqual(expired.estimated_cached, 0)
            self.assertTrue(expired.route["cache_snapshot_expired"])
            router.release(expired, True)
            router.progress(lease, feedback("instance-0", 2, produced=3, age_ms=6000))
            aged = await router.acquire(item("already-aged"))
            self.assertEqual(aged.estimated_cached, 0)
            router.release(aged, True)
            router.progress(lease, feedback("instance-0", 3, produced=4, age_ms=0))
            fresh = await router.acquire(item("fresh"))
            self.assertEqual(fresh.estimated_cached, 768)
            router.release(fresh, True)
            router.release(lease, True)

    async def test_restart_rejects_old_feedback_and_invalidates_hints(self):
        router = self.router()
        old = await router.acquire(item("before-restart"))
        router.progress(old, feedback("instance-0", 8))
        router.observe_backend(0, "replacement-instance")
        with self.assertRaisesRegex(RuntimeError, "restarted"):
            router.progress(old, feedback("instance-0", 9, produced=2))
        router.release(old, False)
        fresh = await router.acquire(item("after-restart"))
        self.assertEqual(fresh.estimated_cached, 0)
        stale = feedback("replacement-instance", 9)
        stale["x_infinilm"]["cache_snapshot"]["backend_instance_id"] = "instance-0"
        router.progress(fresh, stale)
        following = await router.acquire(item("old-snapshot-is-ignored"))
        self.assertEqual(following.estimated_cached, 0)
        router.release(following, True)
        router.progress(fresh, feedback("replacement-instance", 1, produced=2))
        observed = await router.acquire(item("new-instance-epoch-one"))
        self.assertEqual(observed.estimated_cached, 768)
        router.release(observed, True)
        router.release(fresh, True)

    async def test_cancelled_capacity_wait_leaves_no_waiter_or_extra_lease(self):
        router = self.router(max_inflight=1)
        held = await router.acquire(item("held"))
        waiting = asyncio.create_task(router.acquire(item("cancelled")))
        await asyncio.sleep(0)
        self.assertEqual(router.snapshot()["queue_waiters"], 1)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(router.snapshot()["queue_waiters"], 0)
        self.assertEqual(router.snapshot()["active"], [1])
        router.release(held, True)
        self.assertEqual(router.snapshot()["kv_reserved_tokens"], [0])
        self.assertEqual(router.active, {})

    async def test_cancel_after_admission_releases_the_unreturned_lease(self):
        router = self.router(max_inflight=1)
        held = await router.acquire(item("held"))
        waiting = asyncio.create_task(router.acquire(item("grant-cancel-race")))
        await asyncio.sleep(0)
        router.release(held, True)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(router.snapshot()["queue_waiters"], 0)
        self.assertEqual(router.snapshot()["active"], [0])
        self.assertEqual(router.snapshot()["kv_reserved_tokens"], [0])
        self.assertEqual(router.active, {})

    async def test_progress_cannot_refresh_the_v3_wait_deadline(self):
        router = self.router(count=2, max_inflight=1, prefill_slope=1000)
        with clock(1000) as now:
            warm = await router.acquire(item("warm-owner", output=4))
            router.progress(warm, feedback(f"instance-{warm.worker}", 1))
            waiting = asyncio.create_task(router.acquire(item("bounded-wait", output=4)))
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
            now[0] = 1000.1
            router.progress(warm, feedback(f"instance-{warm.worker}", 2, produced=2))
            self.assertFalse(waiting.done())
            now[0] = 1000.193
            router.progress(warm, feedback(f"instance-{warm.worker}", 3, produced=3))
            admitted = await asyncio.wait_for(waiting, 1)
            self.assertNotEqual(admitted.worker, warm.worker)
            evidence = admitted.route["v3_warm_wait"]
            self.assertTrue(evidence["waited_for_warm_copy"])
            self.assertAlmostEqual(evidence["deadline_monotonic"], 1000.192)
            router.release(admitted, True)
            router.release(warm, True)
