# Adapted from the historical V3 experiment for upstream package imports.
# Source: nhcf/InfiniLM@c6f1005f3e81d0799229d66bb6db6c37f1d257ac.
"""Bounded waiting for a reusable full-model replica.

The historical V3 ranking and wait rules are preserved; backend observations
are scoped to the current instance. This upstream adaptation is untested.
"""
import math

from .cache_router import Router as V2Router


class Router(V2Router):
    def __init__(self, backends, policy='cache_bounded_v2', *, warm_wait_ms=1000.0,
                 **kwargs):
        if not math.isfinite(warm_wait_ms) or not 0 <= warm_wait_ms <= 1000:
            raise ValueError('warm_wait_ms must be within [0,1000]')
        self.warm_wait_ms = float(warm_wait_ms)
        self._wait_evidence = {}
        super().__init__(backends, policy, **kwargs)

    def _should_defer(self, ticket, plan, now):
        if plan is None:
            return False
        ident = ticket['item']['id']
        record = self._wait_evidence.setdefault(ident, dict(
            policy='cache_bounded_v3', checks=0, deferred_checks=0,
            first_candidates=[], last_candidates=[], stop_reason=None))
        record['checks'] += 1
        # One deadline per request. Future progress cannot refresh the budget.
        budget = min(self.warm_wait_ms, max(0.0, plan['saved_cold_ms']) * .25)
        deadline = ticket.get('v3_wait_deadline')
        remaining = budget if deadline is None else max(0.0, (deadline-now)*1000)
        record.update(wait_budget_ms=budget, remaining_budget_ms=remaining,
                      original_chosen=plan['chosen'], original_score_ms=plan['scores'][plan['chosen']])
        if not budget or ticket['bypasses'] >= self.max_bypasses or remaining <= 0:
            record['stop_reason'] = ('disabled' if not budget else
                                     'fairness' if ticket['bypasses'] >= self.max_bypasses else 'deadline')
            return False
        p, _, shared, digest, reserve = ticket['contract']
        chosen = plan['chosen']
        margin = max(2.0, plan['saved_cold_ms'] * .05)
        warm_prefill = self.phase_model.prefill(p, shared)
        candidates = []
        for worker, state in enumerate(self.workers):
            if not state.healthy or worker == chosen:
                continue
            # This intervention waits only for a HARD capacity slot. It does
            # not expand the count-bounded set or weaken memory accounting.
            if worker in plan['eligible'] or self._pending(worker):
                continue
            observed = self._cached(worker, digest, shared, now)
            leases = [x for x in self.active.values() if x.worker == worker and x.started
                      and x.backend_instance_id == self.cache[worker]['backend_instance_id']]
            matching = any(x.digest == digest and x.shared >= shared for x in leases)
            if observed < shared and not matching:
                continue
            eta = self._release_eta_ms(worker, ticket['contract'])
            if eta is None or not math.isfinite(eta) or eta < 0:
                continue
            # Earliest departure must actually create enough room for this
            # full reservation. Current experiment uses equal-size leases.
            soonest = min(leases, key=lambda x: max(0, x.output-x.produced))
            fits_after = (state.active_load-1 < self.max_inflight and
                          self.reserved[worker]-soonest.reserved_tokens+reserve <= self.kv_tokens)
            evidence = dict(worker=worker, observed_cached_tokens=observed,
                evidence='fresh_cache_snapshot' if observed >= shared else 'active_matching_decode_prediction',
                release_eta_ms=eta, warm_prefill_ms=warm_prefill,
                projected_ttft_ms=eta+warm_prefill, margin_ms=margin,
                fits_after_first_release=fits_after,
                active_load=state.active_load, pending_requests=0,
                cache_epoch=self.cache[worker]['epoch'],
                cache_age_ms=((now-self.cache[worker]['observed_at'])*1000
                              if self.cache[worker]['observed_at'] > 0 else None),
                decoding=[dict(request_id=x.route['request_id'],
                               digest=x.digest, prompt_tokens=x.prompt, output_tokens=x.output,
                               produced=x.produced, remaining=max(0, x.output-x.produced)) for x in leases])
            evidence['worth_waiting'] = bool(fits_after and eta <= remaining and
                eta+warm_prefill+margin < plan['scores'][chosen])
            candidates.append(evidence)
        record['last_candidates'] = candidates
        suitable = [x for x in candidates if x['worth_waiting']]
        if not suitable:
            record['stop_reason'] = 'no_qualifying_warm_capacity_candidate'
            return False
        if deadline is None:
            deadline = now + budget/1000
            ticket['v3_wait_deadline'] = deadline
            record['first_candidates'] = candidates
            record['wait_started_monotonic'] = now
        record['deadline_monotonic'] = deadline
        record['deferred_checks'] += 1
        record['stop_reason'] = None
        ticket['owner_wait_requested'] = True  # inherited wait accounting field
        ticket['owner_release_eta_ms'] = min(x['release_eta_ms'] for x in suitable)
        self.deferred_decisions += 1
        self._arm_retry(min(deadline, now+.05))
        return True

    def _commit(self, item, contract, eligible, wait_ms, plan):
        lease = super()._commit(item, contract, eligible, wait_ms, plan)
        record = self._wait_evidence.pop(item['id'], None)
        if record is not None:
            record.update(actual_gateway_wait_ms=wait_ms, final_worker=lease.worker,
                          waited_for_warm_copy=record['deferred_checks'] > 0,
                          final_observed_hint=lease.estimated_cached,
                          prediction_is_not_verified_hit=True)
            lease.route['v3_warm_wait'] = record
        lease.route['optimization_version'] = 'cache_bounded_v3'
        return lease

    async def acquire(self, item):
        if getattr(self, '_closed', False):
            raise RuntimeError('The router is closed.')
        try:
            return await super().acquire(item)
        finally:
            self._wait_evidence.pop(item.get('id'), None)

    def snapshot(self):
        result = super().snapshot()
        result.pop('owner_wait_limit_ms', None)
        result.update(policy='cache_bounded_v3', warm_wait_limit_ms=self.warm_wait_ms,
            warm_wait_savings_fraction=.25, warm_wait_margin_fraction=.05,
            warm_wait_changes_count_bound=False, warm_wait_changes_kv_reservation=False,
            warm_wait_requires_no_pending_prefill=True,
            warm_wait_scope='any_healthy_reusable_hard_capacity_blocked_replica')
        return result
