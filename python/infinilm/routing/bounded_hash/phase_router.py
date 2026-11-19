# Adapted from the historical V3 experiment for upstream package imports.
# Source: nhcf/InfiniLM@c6f1005f3e81d0799229d66bb6db6c37f1d257ac.
"""Calibrated phase-pressure routing for the existing full-replica gateway.

This is a candidate ranking proxy, not a GPU simulator or a TTFT guarantee.
Decode contributes ONE co-resident decoding round, never all remaining output.
Cache hints retain the original TTL/pending-Prefill invalidation. Admission still
reserves the full page-rounded input plus declared maximum output on every lease.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path



def _nonnegative(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a finite numeric value')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must be a finite numeric value') from exc
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(name + ' must be finite and ' + ('positive' if positive else 'nonnegative'))
    return value


class PhaseModel:
    """Explicit, immutable calibration; never fitted on benchmark outcomes."""

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            source = json.loads(Path(source).read_text())
        if not isinstance(source, dict) or source.get('schema_version') != 1:
            raise ValueError('phase_model requires schema_version=1')
        self.source = json.loads(json.dumps(source, allow_nan=False))
        coefficients = source.get('prefill_coef')
        if not isinstance(coefficients, (list, tuple)) or len(coefficients) != 3:
            raise ValueError('prefill_coef must contain [a,b,d]')
        self.prefill_coef = tuple(_nonnegative(x, 'prefill_coef') for x in coefficients)
        if not any(self.prefill_coef):
            raise ValueError('prefill_coef must describe a nonzero cost')
        raw_steps = source.get('decode_step_ms')
        if not isinstance(raw_steps, dict) or not raw_steps:
            raise ValueError('decode_step_ms must contain calibrated positive batch keys')
        self.decode_steps = {}
        for key, value in raw_steps.items():
            batch = int(key)
            if str(batch) != str(key) or batch < 1:
                raise ValueError('decode_step_ms keys must be positive integer batch sizes')
            self.decode_steps[batch] = _nonnegative(value, 'decode_step_ms', positive=True)
        self.keys = sorted(self.decode_steps)
        if self.keys != list(range(1, self.keys[-1] + 1)):
            raise ValueError('decode calibration must cover every batch from 1 to max batch')
        contexts = source.get('decode_context_tokens')
        if contexts is None:
            per_request = _nonnegative(source.get('decode_reference_context_tokens'),
                                       'decode_reference_context_tokens', positive=True)
            contexts = {str(k): per_request * k for k in self.keys}
        if not isinstance(contexts, dict):
            raise ValueError('decode_context_tokens must be a batch-to-total-context map')
        self.decode_contexts = {k: _nonnegative(contexts.get(str(k), contexts.get(k)),
                                               'decode_context_tokens', positive=True)
                                for k in self.keys}
        # Measurement noise must not make more decoding requests look cheaper.
        high = 0.0
        for k in self.keys:
            high = max(high, self.decode_steps[k])
            self.decode_steps[k] = high
        self.decode_grid = {}
        raw_grid = source.get('decode_context_samples')
        if raw_grid is not None:
            if not isinstance(raw_grid, dict):
                raise ValueError('decode_context_samples must map batch to samples')
            for batch in self.keys:
                points = raw_grid.get(str(batch), raw_grid.get(batch))
                if not isinstance(points, list) or not points:
                    raise ValueError('Context grid must cover every calibrated decode batch')
                clean = sorted((_nonnegative(x['context_tokens'], 'context sample', positive=True),
                                _nonnegative(x['step_ms'], 'decode step sample', positive=True))
                               for x in points)
                if len({x for x,y in clean}) != len(clean):
                    raise ValueError('Duplicate context calibration point')
                monotone, last = [], 0.0
                for context, step in clean:
                    last = max(last, step)
                    monotone.append((context,last))
                self.decode_grid[batch] = monotone
        self.fingerprint = hashlib.sha256(json.dumps(self.source, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def prefill(self, prompt, cached):
        p = int(prompt)
        if p <= 0:
            raise ValueError('prompt must be positive')
        c = min(max(0, int(cached)), p - 1)
        q = p - c
        a, b, d = self.prefill_coef
        return max(.001, a + b*q/1000 + d*(q*q + 2*q*c)/1e6)

    def decode_step(self, batch, total_context=None):
        if batch == 0:
            return 0.0
        if batch not in self.decode_steps:
            raise ValueError('Uncalibrated decode batch: ' + str(batch))
        context = self.decode_contexts[batch] if total_context is None else _nonnegative(
            total_context, 'active total context')
        if batch in self.decode_grid:
            points = self.decode_grid[batch]
            # Below the smallest measured context retain its fixed FFN cost.
            if context <= points[0][0]:
                return points[0][1]
            for (lo, slo), (hi, shi) in zip(points, points[1:]):
                if context <= hi:
                    return slo + (shi-slo)*(context-lo)/(hi-lo)
            # Only the measured incremental context slope is extrapolated.
            hi, shi = points[-1]
            slope = ((shi-points[-2][1])/(hi-points[-2][0]) if len(points)>1 else 0.0)
            return min(4.0*shi, shi + slope*(context-hi))
        # A simple monotone proxy with bounded extrapolation, explicitly logged.
        ratio = min(4.0, max(.25, context/self.decode_contexts[batch]))
        return self.decode_steps[batch] * ratio
