"""Greedy Qwen MTP using the target engine's built-in head."""

import infinicore
from infinilm.llm.request import MTPRequestState


class MTPRunner:
    def __init__(self, config, target_model_engine):
        hf_config = getattr(target_model_engine, "hf_config", {})
        if isinstance(hf_config, dict):
            text_config = hf_config.get("text_config", hf_config)
            if text_config.get("mtp_num_hidden_layers", 1) != 1:
                raise ValueError("Built-in MTP requires a single MTP layer.")
        self.engine = target_model_engine
        self.block_size = config.block_size
        self.num_draft_tokens = config.num_draft_tokens
        # TP2 trials reduced draft GPU work but regressed end-to-end throughput.
        # Keep the measured batching benefit limited to TP1 for now.
        self.batch_draft = config.tensor_parallel_size == 1
        self.device_tokens = getattr(self.engine, "supports_device_mtp", False)
        self.num_proposals = 0
        self.num_accepted = 0
        self.num_target_calls = 0
        self.num_capacity_fallbacks = 0

    def _target(self, inputs, *, all_positions=False, device_verify=True):
        self.num_target_calls += 1
        return self.engine.forward_raw(
            **inputs,
            sample_all_positions=all_positions,
            return_logits=False,
            verify_draft=all_positions and self.device_tokens and device_verify,
        )

    @staticmethod
    def validate_request(req):
        if req.has_multimodal_inputs or req.sampling_params.top_k != 1:
            raise ValueError(
                "Qwen MTP currently supports text-only greedy requests (`top_k=1`)."
            )
        if (
            not isinstance(req.sampling_params.max_tokens, int)
            or req.sampling_params.max_tokens < 1
        ):
            raise ValueError("Qwen MTP requires a positive integer `max_tokens`.")
        if req.get_prompt_length() < 1:
            raise ValueError("Qwen MTP requires a nonempty tokenized prompt.")

    def _inputs(self, req, tokens, past, *, hidden=None, destinations=None):
        def i32(values):
            return infinicore.from_list(values, dtype=infinicore.int32)

        def i64(values):
            return infinicore.from_list(values, dtype=infinicore.int64)

        device_ids = isinstance(tokens, infinicore.Tensor)
        count = tokens.shape[-1] if device_ids else len(tokens)
        # MTP consumes the next token's embedding with the current position's
        # target hidden state. Its KV slots stay aligned with target slots.
        position_start = past + int(hidden is not None)
        positions = list(range(position_start, position_start + count))
        slots = [
            req.block_table[p // self.block_size] * self.block_size
            + p % self.block_size
            for p in range(past, past + count)
        ]
        return dict(
            input_ids=tokens if device_ids else i64([tokens]),
            position_ids=i64([positions] * self.engine.position_id_axes),
            past_kv_lengths=i32([past]),
            total_kv_lengths=i32([past + count]),
            input_offsets=i32([0, count]),
            cu_seqlens=i32([0, past + count]),
            block_tables=i32([req.block_table]),
            slot_mapping=i64(slots),
            mamba_init_state_indices=i32([0 if past == 0 else req.mamba_cache_index]),
            mamba_final_state_indices=i32(
                [destinations[-1] if destinations else req.mamba_cache_index]
            ),
            token_state_indices=i32(destinations) if destinations else None,
            target_hidden_states=hidden,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )

    @staticmethod
    def _can_continue(req, output):
        remaining = req.sampling_params.max_tokens
        return (
            not req.is_aborted()
            and req.get_num_generated_tokens() + len(output) < remaining
            and (
                req.sampling_params.ignore_eos
                or not any(token in req.eos_token_ids for token in output)
            )
        )

    def _draft(self, req, shifted_tokens, past, hidden, *, commit=True):
        return self._draft_batch([(req, shifted_tokens, past, hidden)], commit=commit)[
            0
        ]

    def forward(self, scheduler_output, model_input):
        requests = scheduler_output.scheduled_requests
        if not requests:
            return []
        if scheduler_output.is_prefill:
            return self._prefill(
                requests, scheduler_output.speculative_cache_ops, model_input
            )
        if len(requests) != 1:
            return self._decode_batch(requests, scheduler_output.speculative_cache_ops)
        req = requests[0]
        cache_ops = scheduler_output.speculative_cache_ops
        past, count, rows, inputs = self._prepare_verify(req, cache_ops)
        if inputs is None:
            return [self._ordinary(req, past, model_input)]
        target = self._target(inputs, all_positions=True)
        expected = list(map(int, target["output_ids"].to_numpy()))
        if self.device_tokens:
            accepted = target["accepted_draft_tokens"]
            output = expected
        else:
            candidates = inputs["input_ids"].to_numpy()[0].tolist()[1:]
            accepted = self._accepted(candidates, expected)
            output = expected[: accepted + 1]
        self._commit(
            req,
            cache_ops,
            past,
            count,
            rows,
            accepted,
            output,
            target["hidden_states"].narrow(1, 0, len(output)),
        )
        return [output]

    @staticmethod
    def _accepted(candidates, expected):
        for index, (candidate, token) in enumerate(zip(candidates, expected)):
            if candidate != token:
                return index
        return len(candidates)

    def _ordinary(self, req, past, inputs=None):
        target = self._target(
            inputs
            if inputs is not None
            else self._inputs(req, [req.generated_token_ids[-1]], past)
        )
        output = [int(target["output_ids"].to_numpy()[-1])]
        if req.mtp_state is not None and self._can_continue(req, output):
            self._draft(req, output, past, target["hidden_states"])
        return output

    def _reserve_verify(self, req, cache_ops):
        self.validate_request(req)
        if cache_ops is None or req.mamba_cache_index is None:
            raise RuntimeError(
                "Qwen MTP requires scheduler-owned paged and recurrent caches."
            )
        past, state = req.get_total_length() - 1, req.mtp_state
        if state is not None and state.cached_tokens != past:
            raise RuntimeError(
                "Qwen MTP draft cache is not aligned with the committed target prefix."
            )
        count = min(
            self.num_draft_tokens,
            req.sampling_params.max_tokens - req.get_num_generated_tokens() - 1,
        )
        if state is None or state.draft_token is None or count < 1:
            return past, 0, None
        try:
            table, _ = cache_ops.append_verify_slots(
                list(req.block_table), req.get_total_length() + 1, count
            )
        except RuntimeError:
            # Allocation checks capacity before mutation. The pending token
            # already has a scheduler-owned slot, so ordinary Decode can run.
            self.num_capacity_fallbacks += 1
            return past, 0, None
        req.block_table, req.num_blocks = table, len(table)
        return past, count, state.scratch_indices[: count + 1]

    def _prepare_verify(self, req, cache_ops):
        past, count, rows = self._reserve_verify(req, cache_ops)
        if not count:
            return past, 0, None, None
        state = req.mtp_state
        proposals, hidden = [state.draft_token], state.draft_hidden
        for step in range(1, count):
            ids = proposals[-1].view((1, 1)) if self.device_tokens else [proposals[-1]]
            # Prefill filled slots `[0, past)`. The next draft writes slot `past`,
            # with position `past+1`; leaving a gap would read unwritten KV.
            token, hidden = self._draft(req, ids, past + step - 1, hidden, commit=False)
            proposals.append(token)
        return past, count, rows, self._verify_inputs(req, past, rows, proposals)

    def _verify_inputs(self, req, past, rows, proposals):
        tokens = [req.generated_token_ids[-1], *proposals]
        if self.device_tokens:
            pending = infinicore.from_list(
                [[req.generated_token_ids[-1]]], dtype=infinicore.int64
            ).to(proposals[0].device)
            tokens = infinicore.cat(
                [pending] + [t.view((1, 1)) for t in proposals], dim=1
            )
        return self._inputs(req, tokens, past, destinations=rows)

    def _commit(
        self, req, cache_ops, past, count, rows, accepted, output, hidden, *, draft=True
    ):
        if not 0 <= accepted <= count or len(output) != accepted + 1:
            raise RuntimeError(
                "MTP verification did not return a valid acceptance length."
            )
        state, old = req.mtp_state, req.mamba_cache_index
        # Select the same checkpoint for Conv and GDN. The correction/bonus
        # token is returned but has no target KV yet.
        req.mamba_cache_index = rows[accepted]
        state.scratch_indices = [
            row for row in (old, *state.scratch_indices) if row != req.mamba_cache_index
        ]
        req.block_table = cache_ops.rollback_to_length(
            req.block_table, past + 1 + accepted
        )
        req.num_blocks, req.slot_mapping = len(req.block_table), []
        continuing = self._can_continue(req, output)
        if continuing and draft:
            self._draft(req, output, past, hidden)
        self.num_proposals += count
        self.num_accepted += accepted
        return continuing

    def _pack(self, items):
        """Pack requests using existing paged metadata, preserving device token IDs."""
        if len(items) == 1:
            return items[0]
        fields = (
            "past_kv_lengths",
            "total_kv_lengths",
            "mamba_init_state_indices",
            "mamba_final_state_indices",
        )
        values = {key: [] for key in fields}
        positions = [[] for _ in range(self.engine.position_id_axes)]
        offsets, cu, slots, tables, destinations = [0], [0], [], [], []
        ids = []
        for item in items:
            ids.append(item["input_ids"])
            offsets.append(offsets[-1] + ids[-1].shape[-1])
            for axis, row in zip(positions, item["position_ids"].to_numpy().tolist()):
                axis.extend(row)
            for key in fields:
                values[key].extend(item[key].to_numpy().tolist())
            cu.append(cu[-1] + values["total_kv_lengths"][-1])
            slots.extend(item["slot_mapping"].to_numpy().tolist())
            tables.extend(item["block_tables"].to_numpy().tolist())
            if item["token_state_indices"] is not None:
                destinations.extend(item["token_state_indices"].to_numpy().tolist())
        width = max(map(len, tables))
        device = ids[0].device
        tokens = infinicore.cat([t.to(device) for t in ids], dim=1)
        result = {
            key: infinicore.from_list(value, dtype=infinicore.int32)
            for key, value in values.items()
        }
        result.update(
            input_ids=tokens,
            position_ids=infinicore.from_list(positions, dtype=infinicore.int64),
            input_offsets=infinicore.from_list(offsets, dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list(cu, dtype=infinicore.int32),
            slot_mapping=infinicore.from_list(slots, dtype=infinicore.int64),
            block_tables=infinicore.from_list(
                [row + [0] * (width - len(row)) for row in tables],
                dtype=infinicore.int32,
            ),
            token_state_indices=infinicore.from_list(
                destinations, dtype=infinicore.int32
            )
            if destinations
            else None,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        if items[0].get("target_hidden_states") is not None:
            result["target_hidden_states"] = infinicore.cat(
                [item["target_hidden_states"] for item in items], dim=1
            )
        return result

    def _draft_batch(self, jobs, *, commit=True):
        """Share the draft's projections across independent requests."""
        if not jobs:
            return []
        if len(jobs) > 1 and not self.batch_draft:
            return [self._draft(*job, commit=commit) for job in jobs]
        inputs = self._pack(
            [
                self._inputs(req, tokens, past, hidden=hidden)
                for req, tokens, past, hidden in jobs
            ]
        )
        result = self.engine.forward_raw(
            **inputs,
            sample_all_positions=False,
            return_logits=False,
            return_device_tokens=self.device_tokens,
        )
        sampled = result["output_ids"]
        if not self.device_tokens:
            sampled = list(map(int, sampled.to_numpy()))
        outputs, offset = [], 0
        for index, (req, tokens, past, _) in enumerate(jobs):
            count = (
                tokens.shape[-1]
                if isinstance(tokens, infinicore.Tensor)
                else len(tokens)
            )
            offset += count
            token = (
                sampled.narrow(0, index, 1) if self.device_tokens else sampled[index]
            )
            hidden = result["hidden_states"].narrow(1, offset - 1, 1)
            if commit:
                req.mtp_state.draft_token = token
                req.mtp_state.draft_hidden = hidden
                req.mtp_state.cached_tokens = past + count
            outputs.append((token, hidden))
        return outputs

    def _prefill(self, requests, cache_ops, model_input):
        for req in requests:
            self.validate_request(req)
            if cache_ops is None or req.mamba_cache_index is None:
                raise RuntimeError(
                    "Qwen MTP requires scheduler-owned paged and recurrent caches."
                )
            if req.num_local_cached_tokens or req.mtp_state is not None:
                raise RuntimeError("Qwen MTP requires a fresh full prompt prefill.")
        target = self._target(model_input)
        sampled = list(map(int, target["output_ids"].to_numpy()))
        offset = 0
        for req, pending in zip(requests, sampled):
            hidden = target["hidden_states"].narrow(1, offset, req.get_prompt_length())
            offset += req.get_prompt_length()
            if self._can_continue(req, [pending]):
                rows = cache_ops.allocate_state_rows(self.num_draft_tokens + 1)
                if rows is not None:
                    req.mtp_state = MTPRequestState(rows)
                    self._draft(
                        req, list(req.prompt_token_ids[1:]) + [pending], 0, hidden
                    )
                else:
                    self.num_capacity_fallbacks += 1
        return [[pending] for pending in sampled]

    def _decode_batch(self, requests, cache_ops):
        outputs, prepared = {}, []
        proposals, hidden = {}, {}
        for req in requests:
            past, count, rows = self._reserve_verify(req, cache_ops)
            if not count:
                outputs[req] = self._ordinary(req, past)
            else:
                prepared.append((req, past, count, rows))
                proposals[req] = [req.mtp_state.draft_token]
                hidden[req] = req.mtp_state.draft_hidden
        for step in range(1, max((item[2] for item in prepared), default=0)):
            active = [item for item in prepared if item[2] > step]
            jobs = [
                (
                    req,
                    proposals[req][-1].view((1, 1))
                    if self.device_tokens
                    else [proposals[req][-1]],
                    past + step - 1,
                    hidden[req],
                )
                for req, past, _, _ in active
            ]
            for (req, *_), (token, state) in zip(
                active, self._draft_batch(jobs, commit=False)
            ):
                proposals[req].append(token)
                hidden[req] = state
        if prepared:
            packed = self._pack(
                [
                    self._verify_inputs(req, past, rows, proposals[req])
                    for req, past, _, rows in prepared
                ]
            )
            target = self._target(packed, all_positions=True, device_verify=False)
            expected = list(map(int, target["output_ids"].to_numpy()))
            candidates = packed["input_ids"].to_numpy()[0].tolist()
            cursor, rebuild = 0, []
            for req, past, count, rows in prepared:
                values = expected[cursor : cursor + count + 1]
                accepted = self._accepted(
                    candidates[cursor + 1 : cursor + count + 1], values
                )
                output = values[: accepted + 1]
                state = target["hidden_states"].narrow(1, cursor, accepted + 1)
                if self._commit(
                    req,
                    cache_ops,
                    past,
                    count,
                    rows,
                    accepted,
                    output,
                    state,
                    draft=False,
                ):
                    rebuild.append((req, output, past, state))
                outputs[req] = output
                cursor += count + 1
            # Different acceptance lengths become variable-length packed
            # requests; their positions, KV tables and hidden states stay separate.
            self._draft_batch(rebuild)
        return [outputs[req] for req in requests]
