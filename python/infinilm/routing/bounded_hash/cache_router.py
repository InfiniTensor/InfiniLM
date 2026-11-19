# Adapted from the historical V3 experiment for upstream package imports.
# Source: nhcf/InfiniLM@c6f1005f3e81d0799229d66bb6db6c37f1d257ac.
"""Cache-preserving bounded hashing for the verified full-replica gateway.

Observed cache hints and predicted reuse are deliberately separate. Predicted
reuse changes ranking only. Every lease reserves its complete input+max-output
sequence, and ordinary pending/TTL invalidation remains in BaseRouter._cached.
"""
import asyncio
from collections import Counter
import math
import time

from .affinity_router import Router as BaseRouter
from .mean_router import Router as MeanRouter
from .phase_router import PhaseModel


class Router(MeanRouter):
    def __init__(self, backends, policy='cache_bounded_v2', *, phase_model,
                 owner_wait_ms=500.0, coalescing=True, **kwargs):
        if policy != 'cache_bounded_v2':
            raise ValueError('Use cache_bounded_v2 with this router')
        if not math.isfinite(owner_wait_ms) or not 0 <= owner_wait_ms <= 2000:
            raise ValueError('owner_wait_ms must be within [0,2000]')
        self.phase_model = PhaseModel(phase_model)
        self.owner_wait_ms = float(owner_wait_ms)
        self.coalescing = bool(coalescing)
        self._retry = None
        self.deferred_decisions = 0
        super().__init__(backends, 'mean_bounded', **kwargs)
        if self.max_inflight > self.phase_model.keys[-1]:
            raise ValueError('Decode calibration must cover the admission cap')
        self.model = self.phase_model

    def affinity_eligible(self, p, o, shared, digest):
        return bool(digest and 0 < shared < p and shared % self.page_size == 0)

    def _forecast(self, worker, contract, observed):
        """Predict serial Prefill reuse; never return it as an observed hit."""
        p, _, shared, digest, _ = contract
        seen = {}
        if self.coalescing:
            for lease in self.active.values():
                if (lease.worker == worker and lease.started and lease.digest and lease.shared
                        and lease.backend_instance_id == self.cache[worker]['backend_instance_id']):
                    seen[lease.digest] = max(seen.get(lease.digest, 0), lease.shared)
        pending_ms = 0.0
        coalesced_pending = 0
        for lease in sorted(self._pending(worker), key=lambda x: x.serial):
            current_instance = (lease.backend_instance_id ==
                                self.cache[worker]['backend_instance_id'])
            predicted = max(lease.estimated_cached,
                            min(lease.shared, seen.get(lease.digest, 0))) if current_instance else 0
            coalesced_pending += predicted > lease.estimated_cached
            pending_ms += self.phase_model.prefill(lease.prompt, predicted)
            if self.coalescing and current_instance and lease.digest and lease.shared:
                seen[lease.digest] = max(seen.get(lease.digest, 0), lease.shared)
        predicted = max(observed, min(shared, seen.get(digest, 0)))
        decode = [x for x in self.active.values() if x.worker == worker and x.started
                  and x.backend_instance_id == self.cache[worker]['backend_instance_id']]
        context = sum(x.prompt + x.produced for x in decode)
        step = self.phase_model.decode_step(len(decode), context)
        own_ms = self.phase_model.prefill(p, predicted)
        return dict(pending_prefill_ms=pending_ms, new_prefill_ms=own_ms,
                    decode_interference_ms=step, active_decode_requests=len(decode),
                    total_decode_context_tokens=context,
                    predicted_cached_tokens=predicted,
                    observed_cached_tokens=observed,
                    coalesced_pending_requests=coalesced_pending,
                    score_ms=pending_ms + own_ms + step)

    def _preview(self, item, contract, eligible, now):
        p, o, shared, digest, _ = contract
        if not self.affinity_eligible(p, o, shared, digest):
            return None
        ring = self._ring_order(item['key'])
        loads = [w.active_load for w in self.workers]
        # Bound projected request count after this admission, not just its score.
        limit = min(self.max_inflight,
                    max(1, math.ceil(self.skew * (sum(loads[i] for i in ring) + 1) / len(ring))))
        bounded = [i for i in ring if i in eligible and loads[i] + 1 <= limit]
        relaxed = not bounded
        if not bounded:
            low = min(loads[i] for i in eligible)
            bounded = [i for i in ring if i in eligible and loads[i] == low]
        observed = {i: self._cached(i, digest, shared, now) for i in bounded}
        components = {i: self._forecast(i, contract, observed[i]) for i in bounded}
        scores = {i: components[i]['score_ms'] for i in bounded}
        owner_key = (digest, shared)
        owners = list(self.owners[owner_key])
        tail = math.ceil((p - shared + o) / self.page_size) * self.page_size
        footprint_limit = self.kv_tokens - self.max_inflight * tail
        visits = Counter(self.recent)[owner_key]
        hot = len(self.recent) >= 16 and visits >= 4 and visits / len(self.recent) >= .15
        order = lambda i: (scores[i], loads[i], self.owner_bytes[i], ring.index(i))
        best = min(bounded, key=order)
        owned = [i for i in owners if i in bounded]
        secondary = None
        if not owned and hot and len(owners) < 2:
            possible = [i for i in bounded if i not in owners
                        and self.owner_bytes[i] + shared <= footprint_limit]
            if possible:
                secondary = min(possible, key=order)
                owned = [secondary]
        resident = [i for i in bounded if observed[i] >= shared]
        stable = list(dict.fromkeys(owned + resident))
        anchor = min(stable, key=order) if stable else None
        chosen = anchor if anchor is not None else best
        # Retention is placement hysteresis, never a claim of a verified hit.
        retention = 0.0
        if anchor in owners:
            predicted = components[anchor]['predicted_cached_tokens']
            retention = max(0.0, self.phase_model.prefill(p, predicted)
                            - self.phase_model.prefill(p, shared))
        effective_anchor = scores[anchor] - retention if anchor is not None else None
        saved_cold_ms = max(0.0, self.phase_model.prefill(p, 0)
                            - self.phase_model.prefill(p, shared))
        escape_margin = max(2.0, saved_cold_ms * .5)
        escaped = (anchor is not None and best != anchor
                   and effective_anchor - scores[best] >= escape_margin
                   and effective_anchor >= 1.5 * scores[best])
        if escaped:
            chosen = best
        commit_secondary = secondary if chosen == secondary else None
        branch = ('cache_hot_secondary' if commit_secondary is not None else
                  'cache_owner' if chosen in owners else
                  'cache_resident_copy' if observed[chosen] >= shared else
                  'cache_overload_escape' if escaped else 'cache_capacity_spill')
        return dict(chosen=chosen, owner_key=owner_key, owners=owners, ring=ring,
                    primary=self.home[owner_key], bounded=bounded, eligible=eligible,
                    projected_count_limit=limit, count_bound_relaxed=relaxed,
                    observed=observed, components=components, scores=scores,
                    best=best, anchor=anchor, retention_credit_ms=retention,
                    escape_margin_ms=escape_margin, branch=branch,
                    commit_secondary=commit_secondary, footprint_limit=footprint_limit,
                    hot=hot, saved_cold_ms=saved_cold_ms)

    def _release_eta_ms(self, worker, contract):
        decoding = [x for x in self.active.values() if x.worker == worker and x.started
                  and x.backend_instance_id == self.cache[worker]['backend_instance_id']]
        if not decoding:
            return None
        component = self._forecast(worker, contract, 0)
        remaining = min(max(0, x.output - x.produced) for x in decoding)
        # Used only to decide whether to wait for a capacity slot. It is not
        # added to the incoming request's TTFT ranking score.
        return component['pending_prefill_ms'] + remaining * component['decode_interference_ms']

    def _should_defer(self, ticket, plan, now):
        if (plan is None or not self.owner_wait_ms or ticket['bypasses'] >= self.max_bypasses
                or any(i in plan['eligible'] for i in plan['owners'])):
            return False
        chosen = plan['chosen']
        shared = ticket['contract'][2]
        if plan['components'][chosen]['predicted_cached_tokens'] >= shared:
            return False
        alternatives = []
        for worker in plan['owners']:
            if not self.workers[worker].healthy:
                continue
            matching = any(x.worker == worker and x.started and x.digest == ticket['contract'][3]
                           for x in self.active.values())
            eta = self._release_eta_ms(worker, ticket['contract']) if matching else None
            if eta is not None and eta <= self.owner_wait_ms:
                warm = self.phase_model.prefill(ticket['contract'][0], shared)
                if eta + warm + 2.0 < plan['scores'][chosen]:
                    alternatives.append(eta)
        if not alternatives:
            return False
        deadline = ticket.setdefault('owner_wait_deadline', now + self.owner_wait_ms / 1000)
        if now >= deadline:
            return False
        self.deferred_decisions += 1
        ticket['owner_wait_requested'] = True
        ticket['owner_release_eta_ms'] = min(alternatives)
        self._arm_retry(min(deadline, now + .05))
        return True

    def _arm_retry(self, when):
        delay = max(.001, when-time.monotonic())
        loop = asyncio.get_running_loop()
        if self._retry is not None:
            if self._retry.when() <= loop.time() + delay:
                return
            self._retry.cancel()
        self._retry = loop.call_later(delay, self._retry_dispatch)

    def _retry_dispatch(self):
        self._retry = None
        self._dispatch()

    def _commit(self, item, contract, eligible, wait_ms, plan):
        if plan is None:
            return BaseRouter._select(self, item, contract, eligible, wait_ms)
        lease = BaseRouter._select(self, item, contract, [plan['chosen']], wait_ms)
        secondary = plan['commit_secondary']
        if secondary is not None:
            self.owners[plan['owner_key']].append(secondary)
            self.owner_bytes[secondary] += contract[2]
        chosen = plan['chosen']
        lease.preferred = plan['primary']
        lease.route.pop('count_limit_before', None)
        lease.route.update(preferred=plan['primary'], spill=chosen != plan['primary'],
            branch=plan['branch'], bounded_candidates=plan['bounded'],
            capacity_candidates=list(eligible), stable_owners=list(self.owners[plan['owner_key']]),
            projected_count_limit=plan['projected_count_limit'],
            count_bound_relaxed=plan['count_bound_relaxed'],
            cached_token_hints=plan['observed'], phase_components=plan['components'],
            scores=plan['scores'], score_units='predicted_serial_prefill_plus_one_decode_ms',
            selected_score_ms=plan['scores'][chosen],
            predicted_cached_tokens=plan['components'][chosen]['predicted_cached_tokens'],
            predicted_reuse_is_verified_hit=False,
            retention_credit_ms=plan['retention_credit_ms'], retention_is_verified_hit=False,
            retention_credit_scope='owner_escape_only',
            secondary_committed=secondary is not None, hot_window=plan['hot'],
            placement_footprint_limit=plan['footprint_limit'],
            escape_margin_ms=plan['escape_margin_ms'],
            phase_model_sha256=self.phase_model.fingerprint,
            decode_rounds_charged=1, decode_remaining_output_tokens_charged=False)
        return lease

    def _dispatch(self):
        if self._dispatching:
            return
        self._dispatching = True
        try:
            while self._waiters:
                healthy = [i for i, w in enumerate(self.workers) if w.healthy]
                if not healthy:
                    while self._waiters:
                        ticket = self._waiters.popleft()
                        if not ticket['future'].done():
                            ticket['future'].set_exception(RuntimeError('No healthy backend'))
                    return
                now = time.monotonic()
                choices = []
                for position, ticket in enumerate(list(self._waiters)[:self.queue_window]):
                    if ticket['future'].done():
                        continue
                    eligible = [i for i in healthy if self.workers[i].active_load < self.max_inflight
                                and self.reserved[i] + ticket['contract'][-1] <= self.kv_tokens]
                    if not eligible:
                        continue
                    plan = self._preview(ticket['item'], ticket['contract'], eligible, now)
                    if self._should_defer(ticket, plan, now):
                        continue
                    score = plan['scores'][plan['chosen']] if plan is not None else float('inf')
                    choices.append((ticket, eligible, plan, score, position))
                if not choices:
                    return
                forced = next((x for x in choices if x[0]['bypasses'] >= self.max_bypasses), None)
                head_choice = next((x for x in choices if x[0] is self._waiters[0]), None)
                selected = forced or (head_choice if head_choice and head_choice[2] is None
                    else min(choices, key=lambda x: (x[3],
                        0 if x[2] is not None and x[2]['chosen'] in x[2]['owners'] else 1, x[4])))
                ticket, eligible, plan, score, position = selected
                for previous in list(self._waiters)[:position]:
                    previous['bypasses'] += 1
                self._waiters.remove(ticket)
                try:
                    lease = self._commit(ticket['item'], ticket['contract'], eligible,
                                         1000*(now-ticket['start']), plan)
                    lease.route.update(queue_policy='cache_bounded_v2_actual_destination',
                        queue_score_ms=score if plan is not None else None,
                        queue_bypasses=ticket['bypasses'], queue_position_on_admission=position,
                        queue_selected_worker=lease.worker,
                        owner_wait_requested=ticket.get('owner_wait_requested', False),
                        owner_release_eta_ms=ticket.get('owner_release_eta_ms'))
                    ticket['future'].set_result(lease)
                except BaseException as exc:
                    ticket['future'].set_exception(exc)
        finally:
            self._dispatching = False
            if not self._waiters and self._retry is not None:
                self._retry.cancel()
                self._retry = None

    def progress(self, lease, obj):
        was_started, produced = lease.started, lease.produced
        super().progress(lease, obj)
        if self._waiters and (was_started != lease.started or produced != lease.produced):
            self._dispatch()

    def close(self):
        if self._retry is not None:
            self._retry.cancel()
            self._retry = None
        while self._waiters:
            ticket = self._waiters.popleft()
            if not ticket['future'].done():
                ticket['future'].set_exception(RuntimeError('The router is closed.'))
        self._closed = True

    def snapshot(self):
        result = super().snapshot()
        for field in ('escape_ratio_threshold', 'escape_prompt_fraction',
                      'retention_credit_scope', 'queue_score_uses_retention_credit'):
            result.pop(field, None)
        result.update(policy='cache_bounded_v2', coalescing=self.coalescing,
            owner_wait_limit_ms=self.owner_wait_ms, deferred_decisions=self.deferred_decisions,
            projected_count_bound='min(max_inflight,ceil(skew*(healthy_active+1)/healthy_workers))',
            prediction_changes_kv_reservation=False, phase_model_sha256=self.phase_model.fingerprint)
        return result
