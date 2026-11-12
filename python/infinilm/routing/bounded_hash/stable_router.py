# Adapted from the historical V3 experiment for upstream package imports.
# Source: nhcf/InfiniLM@c6f1005f3e81d0799229d66bb6db6c37f1d257ac.
"""Bounded stable prefix placement, measured-prefix routing, controlled hot replicas.

No GPU timing/FLOP prediction or training-workload lookahead. The immutable ring
defines placement order. Stable primary ownership is bounded by prefix bytes;
at most one persistent hot secondary is added. Capacity remains conservative.
"""
from collections import Counter, deque
import asyncio
import math
import time
from .affinity_router import Router as BaseRouter


class Router(BaseRouter):
    def __init__(self, backends, policy, **kwargs):
        if policy not in {'least_active', 'round_robin', 'bounded', 'stable_bounded'}:
            raise ValueError('Unknown matrix policy')
        self.matrix_policy = policy
        super().__init__(backends, 'bounded' if policy in {'bounded', 'stable_bounded'} else 'least_active', **kwargs)
        self.owners = {}
        self.owner_bytes = [0] * len(backends)
        self.recent = deque(maxlen=64)
        self.home = {}
        self._dispatching = False
        self.queue_window = 16
        self.max_bypasses = 8

    def _ring_order(self, key):
        return [self.index[w.member_id] for w in self.ring.walk(key) if w.healthy]

    def _register(self, item, contract):
        p,o,shared,digest,_ = contract
        if not self.affinity_eligible(p,o,shared,digest):
            return
        ring=self._ring_order(item['key'])
        if not ring:
            raise RuntimeError('No healthy backend')
        key=(digest,shared)
        if key not in self.owners:
            tail=math.ceil((p-shared+o)/self.page_size)*self.page_size
            limit=min(self.kv_tokens-self.max_inflight*tail,
                      max(shared,math.ceil(self.skew*(sum(self.owner_bytes)+shared)/len(ring)/shared)*shared))
            available=[i for i in ring if self.owner_bytes[i]+shared<=limit]
            primary=available[0] if available else min(ring,key=lambda i:self.owner_bytes[i])
            self.owners[key]=[primary]
            self.owner_bytes[primary]+=shared
            self.home[key]=primary
        self.recent.append(key)

    async def acquire(self, item):
        if self.matrix_policy != 'stable_bounded':
            return await super().acquire(item)
        contract=self._validate_item(item)
        self._register(item,contract)
        future=asyncio.get_running_loop().create_future()
        ticket=dict(item=item,contract=contract,start=time.monotonic(),future=future,bypasses=0)
        self._waiters.append(ticket)
        try:
            self._dispatch()
            return await asyncio.wait_for(asyncio.shield(future),self.queue_timeout_s)
        except (asyncio.TimeoutError,asyncio.CancelledError) as exc:
            if isinstance(exc,asyncio.TimeoutError):
                self.queue_timeouts+=1
            else:
                self.queue_cancellations+=1
            if ticket in self._waiters:
                self._waiters.remove(ticket)
                future.cancel()
            elif future.done() and not future.cancelled() and future.exception() is None:
                lease=future.result()
                if not lease.released:
                    self.release(lease,False)
            if isinstance(exc,asyncio.TimeoutError):
                raise TimeoutError('Gateway affinity queue timeout') from exc
            raise
        finally:
            if ticket in self._waiters:
                self._waiters.remove(ticket)
            self._dispatch()

    def _dispatch(self):
        if self._dispatching:
            return
        self._dispatching=True
        try:
            while self._waiters:
                healthy=[i for i,w in enumerate(self.workers) if w.healthy]
                if not healthy:
                    while self._waiters:
                        self._waiters.popleft()['future'].set_exception(RuntimeError('No healthy backend'))
                    return
                loads=[w.active_load for w in self.workers]
                now=time.monotonic()
                # These experiments have equal full reservations, but retain
                # per-ticket checks for ordinary-request fallback correctness.
                feasible=lambda t:[i for i in healthy if loads[i]<self.max_inflight
                                   and self.reserved[i]+t['contract'][-1]<=self.kv_tokens]
                window=list(self._waiters)[:self.queue_window]
                candidates=[t for t in window if feasible(t)]
                if not candidates:
                    return
                head=self._waiters[0]
                forced=next((t for t in candidates if t['bypasses']>=self.max_bypasses),None)
                def priority(t):
                    p,o,shared,digest,_=t['contract']
                    if not self.affinity_eligible(p,o,shared,digest):
                        return (float('inf'),1,window.index(t))
                    eligible=feasible(t)
                    costs=[sum(x.pending_tokens for x in self._pending(i))+p-self._cached(i,digest,shared,now)
                           for i in eligible]
                    owned=any(i in eligible for i in self.owners.get((digest,shared),()))
                    return (min(costs),0 if owned else 1,window.index(t))
                p,o,shared,digest,_=head['contract']
                ordinary=not self.affinity_eligible(p,o,shared,digest)
                ticket=forced or (head if ordinary and head in candidates else min(candidates,key=priority))
                position=list(self._waiters).index(ticket)
                for previous in list(self._waiters)[:position]:
                    previous['bypasses']+=1
                self._waiters.remove(ticket)
                try:
                    lease=self._select(ticket['item'],ticket['contract'],feasible(ticket),1000*(now-ticket['start']))
                    lease.route.update(queue_policy='bounded_affinity_window',queue_bypasses=ticket['bypasses'],
                                       queue_position_on_admission=position)
                    ticket['future'].set_result(lease)
                except BaseException as exc:
                    ticket['future'].set_exception(exc)
        finally:
            self._dispatching=False

    def release(self, lease, success, usage=None):
        super().release(lease,success,usage)
        if self.matrix_policy=='stable_bounded':
            self._dispatch()

    def _select(self, item, contract, eligible, wait_ms):
        if self.matrix_policy in {'least_active', 'bounded'}:
            return super()._select(item, contract, eligible, wait_ms)
        if self.matrix_policy == 'round_robin':
            n = len(self.workers)
            chosen = next(i for i in ((self.turn+j) % n for j in range(n)) if i in eligible)
            lease = super()._select(item, contract, [chosen], wait_ms)
            lease.preferred = chosen
            lease.route.update(preferred=chosen, branch='round_robin_capacity_eligible', spill=False)
            return lease
        p, o, shared, digest, reserve = contract
        if not self.affinity_eligible(p, o, shared, digest):
            return super()._select(item, contract, eligible, wait_ms)

        ring = self._ring_order(item['key'])
        loads = [w.active_load for w in self.workers]
        now = time.monotonic()
        # Reserve room for three private suffixes, independently of any hint.
        tail = math.ceil((p-shared+o)/self.page_size) * self.page_size
        footprint_limit = self.kv_tokens-self.max_inflight*tail
        owner_key = (digest, shared)
        owners = self.owners[owner_key]
        visits = Counter(self.recent)[owner_key]
        hot = len(self.recent) >= 16 and visits >= 4 and visits/len(self.recent) >= .15

        mean = sum(loads[i] for i in ring)/len(ring)
        bounded = [i for i in ring if i in eligible and loads[i] <= self.skew*mean]
        if not bounded:
            low = min(loads[i] for i in eligible)
            bounded = [i for i in ring if i in eligible and loads[i] == low]
        cached = {i:self._cached(i, digest, shared, now) for i in bounded}
        pending = {i:sum(x.pending_tokens for x in self._pending(i)) for i in bounded}
        scores = {i:pending[i]+p-cached[i] for i in bounded}
        resident = [i for i in bounded if cached[i] >= shared]
        owned = [i for i in owners if i in bounded]
        new_secondary = False
        # Only persistent hot load expands the owner set. Ordinary spills do
        # not turn into permanent replicas and displace all other documents.
        if not owned and hot and len(owners) < 2:
            candidates = [i for i in bounded if i not in owners
                          and self.owner_bytes[i]+shared <= footprint_limit]
            if candidates:
                replica = min(candidates, key=lambda i:(scores[i],self.owner_bytes[i],loads[i],ring.index(i)))
                owners.append(replica)
                self.owner_bytes[replica] += shared
                owned = [replica]
                new_secondary = True
        # Keep the stable owner unless a known resident copy has lower prefill
        # work. Never make a speculative cold replica merely to win a small
        # instantaneous queue-count difference.
        candidates = list(dict.fromkeys(owned+resident))
        if candidates:
            chosen = min(candidates, key=lambda i:(scores[i],loads[i],owners.index(i) if i in owners else 99,ring.index(i)))
            branch = 'stable_hot_secondary' if new_secondary else 'stable_owner' if chosen in owners else 'stable_resident_copy'
        else:
            chosen = min(bounded, key=lambda i:(scores[i],self.owner_bytes[i],loads[i],ring.index(i)))
            branch = 'stable_capacity_spill'
        primary = self.home[owner_key]
        lease = super()._select(item, contract, [chosen], wait_ms)
        lease.preferred = primary
        lease.route.update(preferred=primary, spill=chosen != primary, branch=branch,
                           ring_primary=ring[0], stable_owners=list(owners), hot_window=hot,
                           owner_prefix_tokens=list(self.owner_bytes),
                           bounded_candidates=bounded, cached_token_hints=cached,
                           pending_uncached_tokens=pending, scores=scores,
                           score_units='uncached_tokens', candidate_prefill_cost={i:p-cached[i] for i in bounded},
                           placement_footprint_limit=footprint_limit)
        return lease

    def snapshot(self):
        result = super().snapshot()
        result.update(policy=self.matrix_policy, owner_prefix_tokens=list(self.owner_bytes),
                      owned_prefixes=len(self.owners), persistent_replicas=sum(map(len,self.owners.values())),
                      affinity_queue_window=self.queue_window,max_queue_bypasses=self.max_bypasses)
        return result
