# Adapted from the historical V3 experiment for upstream package imports.
# Source: nhcf/InfiniLM@c6f1005f3e81d0799229d66bb6db6c37f1d257ac.
"""Stable bounded routing with a guarded owner escape and consistent queue scores.

Scores count outstanding uncached input tokens. They are a conservative heuristic,
not calibrated milliseconds or a prediction of GPU completion time. A routing-only
retention credit conservatively resists duplicating a potentially retained owner
prefix; it is not a verified hit and never changes cache hints or KV reservations.
Legacy policies delegate unchanged to stable_router.Router.
"""
from collections import Counter
import math
import time

from .affinity_router import Router as BaseRouter
from .stable_router import Router as StableRouter


class Router(StableRouter):
    def __init__(self, backends, policy, **kwargs):
        self.mean_enabled = policy == 'mean_bounded'
        # Keep the inherited stable-policy acquisition/cancellation/lease lifecycle.
        super().__init__(backends, 'stable_bounded' if self.mean_enabled else policy, **kwargs)
        self.escape_ratio = 1.5
        self.escape_prompt_fraction = .5

    def _preview(self, item, contract, eligible, now):
        """Propose an admission without changing owners, leases, or queue state."""
        p, o, shared, digest, _ = contract
        if not self.affinity_eligible(p, o, shared, digest):
            return None
        ring = self._ring_order(item['key'])
        loads = [w.active_load for w in self.workers]
        mean = sum(loads[i] for i in ring) / len(ring)
        bounded = [i for i in ring if i in eligible and loads[i] <= self.skew * mean]
        if not bounded:
            low = min(loads[i] for i in eligible)
            bounded = [i for i in ring if i in eligible and loads[i] == low]
        cached = {i: self._cached(i, digest, shared, now) for i in bounded}
        pending = {i: sum(x.pending_tokens for x in self._pending(i)) for i in bounded}
        scores = {i: pending[i] + p - cached[i] for i in bounded}
        owner_key = (digest, shared)
        owners = list(self.owners[owner_key])
        owned = [i for i in owners if i in bounded]
        resident = [i for i in bounded if cached[i] >= shared]
        tail = math.ceil((p - shared + o) / self.page_size) * self.page_size
        footprint_limit = self.kv_tokens - self.max_inflight * tail
        visits = Counter(self.recent)[owner_key]
        hot = len(self.recent) >= 16 and visits >= 4 and visits / len(self.recent) >= .15
        global_key = lambda i: (scores[i], self.owner_bytes[i], loads[i], ring.index(i))
        proposed_secondary = None
        if not owned and hot and len(owners) < 2:
            possible = [i for i in bounded if i not in owners
                        and self.owner_bytes[i] + shared <= footprint_limit]
            if possible:
                proposed_secondary = min(possible, key=global_key)
                owned = [proposed_secondary]
        stable_candidates = list(dict.fromkeys(owned + resident))
        global_best = min(bounded, key=global_key)
        stable_best = None
        if stable_candidates:
            stable_best = min(stable_candidates, key=lambda i: (
                scores[i], loads[i], owners.index(i) if i in owners else 99, ring.index(i)))
            chosen = stable_best
            branch = ('mean_hot_secondary' if chosen == proposed_secondary else
                      'mean_owner' if chosen in owners else 'mean_resident_copy')
        else:
            chosen, branch = global_best, 'mean_capacity_spill'
        # Pending prefills conservatively erase cache hints. Before abandoning
        # an established owner, credit its possibly retained prefix ONLY in the
        # escape comparison. This is resistance to cold duplication, not a cache
        # hit prediction. Raw scores still drive queue order and reported hints.
        retention_credit = (max(0, shared-cached[stable_best])
                            if stable_best is not None and stable_best in owners else 0)
        effective_stable_score = (scores[stable_best]-retention_credit
                                  if stable_best is not None else None)
        # Both a relative and an absolute advantage are needed to leave an owner.
        escape_margin = max(p * self.escape_prompt_fraction, shared / 2)
        if (stable_best is not None and global_best != stable_best
                and effective_stable_score - scores[global_best] >= escape_margin
                and effective_stable_score >= self.escape_ratio * scores[global_best]):
            chosen, branch = global_best, 'mean_overload_escape'
        # A proposal that loses the final comparison must not consume footprint.
        commit_secondary = proposed_secondary if chosen == proposed_secondary else None
        return dict(chosen=chosen, branch=branch, owner_key=owner_key, ring=ring,
                    bounded=bounded, cached=cached, pending=pending, scores=scores,
                    stable_best=stable_best, global_best=global_best,
                    commit_secondary=commit_secondary, hot=hot,
                    footprint_limit=footprint_limit, escape_margin=escape_margin,
                    retention_credit=retention_credit, effective_stable_score=effective_stable_score,
                    count_limit=self.skew * mean)

    def _commit(self, item, contract, eligible, wait_ms, plan):
        if plan is None:
            return BaseRouter._select(self, item, contract, eligible, wait_ms)
        chosen = plan['chosen']
        p, _, shared, _, _ = contract
        owners = self.owners[plan['owner_key']]
        secondary = plan['commit_secondary']
        if secondary is not None and (secondary in owners or len(owners) >= 2):
            raise RuntimeError('Preview/commit owner state changed during synchronous admission')
        # Validate before admission so a rejected preview cannot strand a lease.
        lease = BaseRouter._select(self, item, contract, [chosen], wait_ms)
        if secondary is not None:
            owners.append(secondary)
            self.owner_bytes[secondary] += shared
        primary = self.home[plan['owner_key']]
        lease.preferred = primary
        stable = plan['stable_best']
        lease.route.update(
            preferred=primary, spill=chosen != primary, branch=plan['branch'],
            ring_primary=plan['ring'][0], stable_owners=list(owners),
            hot_window=plan['hot'], owner_prefix_tokens=list(self.owner_bytes),
            bounded_candidates=plan['bounded'], cached_token_hints=plan['cached'],
            pending_uncached_tokens=plan['pending'], scores=plan['scores'],
            score_units='uncached_tokens_heuristic',
            candidate_prefill_cost={i: p-plan['cached'][i] for i in plan['bounded']},
            placement_footprint_limit=plan['footprint_limit'],
            count_limit_before=plan['count_limit'],
            comparison_margin=plan['escape_margin'],
            hash_fallback=stable if stable is not None else primary,
            best_candidate=plan['global_best'],
            escape_ratio_threshold=self.escape_ratio,
            escape_margin_tokens=plan['escape_margin'],
            stable_candidate=stable,
            stable_score_tokens=plan['scores'][stable] if stable is not None else None,
            routing_only_retention_credit_tokens=plan['retention_credit'],
            effective_stable_score_tokens=plan['effective_stable_score'],
            retention_credit_is_verified_hit=False,
            queue_score_uses_retention_credit=False,
            best_score_tokens=plan['scores'][plan['global_best']],
            selected_score_tokens=plan['scores'][chosen],
            selected_cached_hint=plan['cached'][chosen],
            secondary_committed=secondary is not None)
        return lease

    def _select(self, item, contract, eligible, wait_ms):
        if not self.mean_enabled:
            return super()._select(item, contract, eligible, wait_ms)
        return self._commit(item, contract, eligible, wait_ms,
                            self._preview(item, contract, eligible, time.monotonic()))

    def _dispatch(self):
        if not self.mean_enabled:
            return super()._dispatch()
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
                loads = [w.active_load for w in self.workers]
                window = list(self._waiters)[:self.queue_window]
                choices = []
                for position, ticket in enumerate(window):
                    if ticket['future'].done():
                        continue
                    eligible = [i for i in healthy if loads[i] < self.max_inflight
                                and self.reserved[i]+ticket['contract'][-1] <= self.kv_tokens]
                    if eligible:
                        plan = self._preview(ticket['item'], ticket['contract'], eligible, now)
                        score = plan['scores'][plan['chosen']] if plan is not None else float('inf')
                        choices.append((ticket, eligible, plan, score, position))
                if not choices:
                    return
                forced = next((x for x in choices if x[0]['bypasses'] >= self.max_bypasses), None)
                head = self._waiters[0]
                head_choice = next((x for x in choices if x[0] is head), None)
                # On equal actual costs, prefer an established owner. Pending
                # prefills can hide a retained prefix; ownership is only a
                # tie-break, never a verified hit or a reduced KV reservation.
                selected = forced or (head_choice if head_choice and head_choice[2] is None
                                      else min(choices, key=lambda x: (
                                          x[3],
                                          0 if x[2] is not None and x[2]['chosen'] in
                                          self.owners.get(x[2]['owner_key'], ()) else 1,
                                          x[4])))
                ticket, eligible, plan, score, position = selected
                for previous in list(self._waiters)[:position]:
                    previous['bypasses'] += 1
                self._waiters.remove(ticket)
                try:
                    # No await or state mutation occurs between preview and this
                    # commit; the queue score refers to the worker actually used.
                    lease = self._commit(ticket['item'], ticket['contract'], eligible,
                                         1000*(now-ticket['start']), plan)
                    lease.route.update(queue_policy='mean_bounded_actual_destination',
                                       queue_owner_tiebreak='2026-10-22.1',
                                       queue_bypasses=ticket['bypasses'],
                                       queue_position_on_admission=position,
                                       queue_selected_worker=lease.worker,
                                       queue_score_tokens=score if plan is not None else None)
                    ticket['future'].set_result(lease)
                except BaseException as exc:
                    ticket['future'].set_exception(exc)
        finally:
            self._dispatching = False

    def snapshot(self):
        result = super().snapshot()
        if self.mean_enabled:
            result.update(policy='mean_bounded', escape_ratio_threshold=self.escape_ratio,
                          queue_owner_tiebreak='2026-10-22.1',
                          escape_prompt_fraction=self.escape_prompt_fraction,
                          queue_scores_actual_destination=True,
                          retention_credit_scope='escape_decision_only_not_verified_hit',
                          queue_score_uses_retention_credit=False)
        return result
