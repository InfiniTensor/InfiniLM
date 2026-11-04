import logging
import weakref

from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
from infinilm.infer_engine import read_hf_config
from infinilm.modeling_utils import load_model_state_dict_by_file

from ... import _core as infinicore

logger = logging.getLogger(__name__)


class SpeculativeRunner:
    def __new__(cls, config, target_model_engine, device):
        draft_config = read_hf_config(config.draft_model_path)
        runner = (
            DSparkSpeculativeRunner
            if draft_config.get("model_type") == "qwen3_dspark"
            else EagleSpeculativeRunner
        )
        return runner(config, target_model_engine, device, draft_config)


class EagleSpeculativeRunner:
    def __init__(self, config, target_model_engine, device, draft_config=None):
        self.config = config
        self.target_model_engine = target_model_engine
        self.num_draft_tokens = config.num_draft_tokens
        self.draft_max_batch_size = config.max_batch_size
        self.eagle_accept_count = 0
        self.eagle_total_count = 0

        draft_cache_config = StaticKVCacheConfig(
            max_batch_size=config.max_batch_size, max_cache_len=config.max_cache_len
        )
        self.draft_model_engine = target_model_engine.attach_draft(
            config.draft_model_path,
            cache_config=draft_cache_config,
            attention_backend="default",
            use_mla=False,
        )
        if self.draft_model_engine.model_type != "minicpm_eagle":
            raise RuntimeError(
                f"draft_model_path must point to a MiniCPM Eagle draft model, "
                f"got model_type={self.draft_model_engine.model_type}"
            )
        if not config.skip_load:
            load_model_state_dict_by_file(
                self.draft_model_engine,
                config.draft_model_path,
                dtype=self.draft_model_engine.dtype,
            )

    def forward(self, scheduler_output, model_input):
        cache_ops = getattr(scheduler_output, "speculative_cache_ops", None)
        if cache_ops is None:
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        requests = scheduler_output.scheduled_requests
        if not requests:
            return []

        target_model_input = dict(model_input)
        target_model_input["sample_all_positions"] = scheduler_output.is_prefill
        target_output = self.target_model_engine.forward_raw(**target_model_input)
        target_token_ids = target_output["output_ids"].to_numpy().tolist()
        if not target_token_ids:
            return target_token_ids

        input_offsets = model_input["input_offsets"].to_numpy().tolist()
        hidden_states = target_output["hidden_states"]
        output_tokens_by_req: list[list[int]] = [[] for _ in requests]
        draft_jobs = []

        for req_idx, req in enumerate(requests):
            last_input_idx = int(input_offsets[req_idx + 1]) - 1
            target_token = int(target_token_ids[last_input_idx])
            max_tokens = req.sampling_params.max_tokens
            remaining = (
                None
                if max_tokens is None
                else max_tokens - req.get_num_generated_tokens()
            )
            if remaining is not None and remaining <= 1:
                output_tokens_by_req[req_idx] = [target_token]
                continue

            draft_budget = self.num_draft_tokens
            if remaining is not None:
                draft_budget = min(draft_budget, max(1, remaining - 1))
            if draft_budget <= 0:
                output_tokens_by_req[req_idx] = [target_token]
                continue

            source_token, source_position = self._get_last_input_token_and_position(
                req, scheduler_output.is_prefill
            )
            draft_jobs.append(
                {
                    "req_idx": req_idx,
                    "req": req,
                    "target_token": target_token,
                    "remaining": remaining,
                    "source_token": source_token,
                    "source_position": source_position,
                    "target_hidden": hidden_states.narrow(1, last_input_idx, 1),
                    "num_tokens": draft_budget,
                }
            )

        draft_results = self._draft_eagle_tokens_batch(draft_jobs)
        verify_candidates = []
        for job, draft_tokens in zip(draft_jobs, draft_results):
            req_idx = job["req_idx"]
            req = job["req"]
            target_token = job["target_token"]
            if not draft_tokens:
                output_tokens_by_req[req_idx] = [target_token]
                continue

            self.eagle_total_count += len(draft_tokens)
            if draft_tokens[0] != target_token:
                output_tokens_by_req[req_idx] = [target_token]
                continue

            verify_candidates.append(
                {
                    "req_idx": req_idx,
                    "req": req,
                    "base_len": req.get_total_length(),
                    "remaining": job["remaining"],
                    "draft_tokens": draft_tokens,
                }
            )

        if verify_candidates:
            with cache_ops.verify_batch(verify_candidates) as reservation:
                verify_input = self._build_paged_verify_batch_input(verify_candidates)
                reservation.mark_submitted()
                verify_output = self.target_model_engine.forward_raw(**verify_input)
                verify_token_ids = verify_output["output_ids"].to_numpy().tolist()
                reservation.mark_completed()
                expected_count = sum(
                    len(row["draft_tokens"]) for row in verify_candidates
                )
                if len(verify_token_ids) != expected_count:
                    raise RuntimeError(
                        "verification output count does not match the batch"
                    )
                if any(
                    type(token) is not int or token < 0 for token in verify_token_ids
                ):
                    raise RuntimeError(
                        "verification output requires non-negative integer token IDs"
                    )
                keep_lengths = []
                accepted_total = 0
                offset = 0
                for candidate in verify_candidates:
                    req = candidate["req"]
                    draft_tokens = candidate["draft_tokens"]
                    segment = verify_token_ids[offset : offset + len(draft_tokens)]
                    offset += len(draft_tokens)
                    accepted = 1
                    correction = None
                    for draft_idx in range(1, len(draft_tokens)):
                        expected = segment[draft_idx - 1]
                        if draft_tokens[draft_idx] != expected:
                            correction = expected
                            break
                        accepted += 1
                    if correction is None:
                        correction = segment[-1]
                    output_tokens = draft_tokens[:accepted] + [correction]
                    remaining = candidate["remaining"]
                    if remaining is not None:
                        output_tokens = output_tokens[:remaining]
                    if not req.sampling_params.ignore_eos:
                        for index, token in enumerate(output_tokens):
                            if token in req.eos_token_ids:
                                output_tokens = output_tokens[: index + 1]
                                break
                    accepted = min(accepted, len(output_tokens))
                    accepted_total += accepted
                    keep_lengths.append(candidate["base_len"] + accepted)
                    output_tokens_by_req[candidate["req_idx"]] = output_tokens
                reservation.commit(keep_lengths)
                self.eagle_accept_count += accepted_total

        return output_tokens_by_req

    def _get_last_input_token_and_position(self, req, is_prefill):
        if is_prefill:
            return req.prompt_token_ids[-1], req.prompt_length - 1
        token = (
            req.generated_token_ids[-1]
            if req.generated_token_ids
            else req.prompt_token_ids[-1]
        )
        return token, req.get_total_length() - 1

    def _draft_eagle_tokens_batch(self, jobs: list[dict]) -> list[list[int]]:
        if not jobs:
            return []

        draft_tokens_by_job: list[list[int]] = [[] for _ in jobs]
        current_tokens = [int(job["source_token"]) for job in jobs]
        current_hiddens = [job["target_hidden"] for job in jobs]
        max_steps = max(int(job["num_tokens"]) for job in jobs)
        if max_steps <= 0:
            return draft_tokens_by_job

        real_batch = len(jobs)
        draft_batch = max(self.draft_max_batch_size, real_batch)
        if real_batch > self.draft_max_batch_size:
            raise RuntimeError(
                f"Eagle draft batch {real_batch} exceeds configured max_batch_size "
                f"{self.draft_max_batch_size}. Increase max_batch_size when creating LLM."
            )
        dummy_token = current_tokens[0]
        dummy_hidden = current_hiddens[0]

        for step in range(max_steps):
            input_tokens = [
                current_tokens[idx] if idx < real_batch else dummy_token
                for idx in range(draft_batch)
            ]
            positions = [
                int(jobs[idx]["source_position"]) + step if idx < real_batch else 0
                for idx in range(draft_batch)
            ]
            hidden_inputs = [
                current_hiddens[idx] if idx < real_batch else dummy_hidden
                for idx in range(draft_batch)
            ]
            target_hidden = infinicore.cat(hidden_inputs, dim=0)
            seq_len = step + 1

            draft_output = self.draft_model_engine.forward_raw(
                input_ids=infinicore.from_list(
                    [[token] for token in input_tokens], dtype=infinicore.int64
                ),
                position_ids=infinicore.from_list(
                    [[pos] for pos in positions], dtype=infinicore.int64
                ),
                past_kv_lengths=infinicore.from_list(
                    [step] * draft_batch, dtype=infinicore.int32
                ),
                total_kv_lengths=infinicore.from_list(
                    [seq_len] * draft_batch, dtype=infinicore.int32
                ),
                input_offsets=infinicore.from_list(
                    list(range(draft_batch + 1)), dtype=infinicore.int32
                ),
                cu_seqlens=infinicore.from_list(
                    [i * seq_len for i in range(draft_batch + 1)],
                    dtype=infinicore.int32,
                ),
                target_hidden_states=target_hidden,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                sample_all_positions=False,
            )
            token_ids = draft_output["output_ids"].to_numpy().tolist()
            draft_hidden = draft_output["hidden_states"]
            for job_idx, job in enumerate(jobs):
                token = int(token_ids[job_idx])
                if step < int(job["num_tokens"]):
                    draft_tokens_by_job[job_idx].append(token)
                current_tokens[job_idx] = token
                current_hiddens[job_idx] = draft_hidden.narrow(0, job_idx, 1)

        return draft_tokens_by_job

    def _build_paged_verify_batch_input(self, candidates: list[dict]) -> dict:
        tokens = []
        position_ids = []
        past_lens = []
        seq_lens = []
        input_offsets = [0]
        cu_seqlens = [0]
        slot_mapping = []
        block_tables = []
        max_block_table_len = max(
            len(candidate["block_table"]) for candidate in candidates
        )
        # A one-token candidate only needs its correction token. Selecting the
        # last position keeps that case eligible for the compiled decode graph;
        # longer candidates still need every intermediate prediction.
        sample_all_positions = any(
            len(candidate["draft_tokens"]) != 1 for candidate in candidates
        )

        for candidate in candidates:
            base_len = candidate["base_len"]
            draft_tokens = candidate["draft_tokens"]
            tokens.extend(draft_tokens)
            position_ids.extend(range(base_len, base_len + len(draft_tokens)))
            past_lens.append(base_len)
            seq_lens.append(base_len + len(draft_tokens))
            input_offsets.append(input_offsets[-1] + len(draft_tokens))
            cu_seqlens.append(cu_seqlens[-1] + base_len + len(draft_tokens))
            slot_mapping.extend(candidate["slot_mapping"])
            block_tables.append(
                candidate["block_table"]
                + [-1] * (max_block_table_len - len(candidate["block_table"]))
            )

        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(past_lens, dtype=infinicore.int32),
            "total_kv_lengths": infinicore.from_list(seq_lens, dtype=infinicore.int32),
            "input_offsets": infinicore.from_list(
                input_offsets, dtype=infinicore.int32
            ),
            "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_all_positions": sample_all_positions,
        }


class DSparkSpeculativeRunner(EagleSpeculativeRunner):
    """Fixed-width greedy DSpark using the paged speculative reservation path."""

    def __init__(self, config, target_model_engine, device, draft_config):
        if config.cache_type != "paged":
            raise ValueError("Qwen3 DSpark requires cache_type='paged'")
        if config.enable_prefix_caching:
            raise ValueError(
                "Qwen3 DSpark does not yet share prefix-cache ownership; "
                "launch with --disable-prefix-caching"
            )
        if config.tensor_parallel_size != 1 or config.pipeline_parallel_size != 1:
            raise ValueError("initial Qwen3 DSpark support requires TP=1 and PP=1")
        if target_model_engine.model_type != "qwen3":
            raise ValueError(
                "Qwen3 DSpark requires a qwen3 target model, got "
                f"{target_model_engine.model_type}"
            )

        self.config = config
        self.target_model_engine = target_model_engine
        self.draft_config = draft_config
        self.verify_num_draft_tokens = int(config.num_draft_tokens)
        if self.verify_num_draft_tokens < 2:
            raise ValueError(
                "Qwen3 DSpark requires num_draft_tokens >= 2 (gamma + anchor)"
            )
        self.num_draft_tokens = self.verify_num_draft_tokens - 1
        checkpoint_gamma = draft_config.get("block_size")
        if (
            checkpoint_gamma is not None
            and int(checkpoint_gamma) != self.num_draft_tokens
        ):
            logger.warning(
                "DSpark gamma mismatch: using gamma=%s from num_draft_tokens=%s, "
                "but the checkpoint block_size is %s",
                self.num_draft_tokens,
                self.verify_num_draft_tokens,
                checkpoint_gamma,
            )
        self.target_layer_ids = [int(x) for x in draft_config["target_layer_ids"]]
        self.mask_token_id = int(draft_config["mask_token_id"])
        self.eagle_accept_count = 0
        self.eagle_total_count = 0
        self._stateful_requests = weakref.WeakSet()
        self._pinned_metadata = {}
        self._pending_target_hidden = None
        self._verify_metadata_slot = 0

        draft_cache_config = PagedKVCacheConfig(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            max_batch_size=config.max_batch_size,
        )
        self.draft_model_engine = target_model_engine.attach_draft(
            config.draft_model_path,
            cache_config=draft_cache_config,
            attention_backend=config.attn_backend,
            use_mla=False,
            config_overrides={
                "speculative_verify_width": self.num_draft_tokens,
                "speculative_markov_fused": True,
                "speculative_cache_injection_width": self.verify_num_draft_tokens,
            },
        )
        if self.draft_model_engine.model_type != "qwen3_dspark":
            raise RuntimeError(
                "draft_model_path must point to Qwen3DSparkModel, got "
                f"model_type={self.draft_model_engine.model_type}"
            )
        if not config.skip_load:
            load_model_state_dict_by_file(
                self.draft_model_engine,
                config.draft_model_path,
                dtype=self.draft_model_engine.dtype,
            )

    def forward(self, scheduler_output, model_input):
        cache_ops = getattr(scheduler_output, "speculative_cache_ops", None)
        if cache_ops is None:
            return self.target_model_engine.forward(**model_input).to_numpy().tolist()

        requests = scheduler_output.scheduled_requests
        if not requests:
            return []
        if any(req.sampling_params.top_k != 1 for req in requests):
            raise NotImplementedError(
                "Qwen3 DSpark currently supports exact greedy decoding only (top_k=1)"
            )

        remaining_by_req = []
        for req in requests:
            max_tokens = req.sampling_params.max_tokens
            remaining_by_req.append(
                None
                if max_tokens is None
                else max_tokens - req.get_num_generated_tokens()
            )
        device_token_path = all(
            remaining is None or remaining >= self.verify_num_draft_tokens
            for remaining in remaining_by_req
        )

        stateful_device_cycle = (
            device_token_path
            and not scheduler_output.is_prefill
            and all(req in self._stateful_requests for req in requests)
        )
        if self._pending_target_hidden is not None and not (
            stateful_device_cycle and self._pending_matches(requests)
        ):
            self._flush_pending_target_hidden()

        if stateful_device_cycle:
            return self._forward_stateful_device_token_cycle(
                scheduler_output,
                model_input,
                remaining_by_req,
                cache_ops,
            )

        target_input = dict(model_input)
        target_input["sample_all_positions"] = False
        target_input["capture_layer_ids"] = self.target_layer_ids
        target_input["output_ids_on_device"] = device_token_path
        target_output = self.target_model_engine.forward_raw(**target_input)
        if device_token_path:
            result = self._forward_device_token_cycle(
                scheduler_output,
                model_input,
                target_output,
                remaining_by_req,
                cache_ops,
            )
            self._stateful_requests.update(requests)
            return result

        anchors = [
            int(token) for token in target_output["output_ids"].to_numpy().tolist()
        ]
        if len(anchors) != len(requests):
            raise RuntimeError(
                "DSpark target output count does not match request count"
            )

        output_tokens_by_req: list[list[int]] = [[] for _ in requests]
        jobs = []
        for req_idx, (req, anchor) in enumerate(zip(requests, anchors)):
            max_tokens = req.sampling_params.max_tokens
            remaining = (
                None
                if max_tokens is None
                else max_tokens - req.get_num_generated_tokens()
            )
            if remaining is not None and remaining <= 1:
                output_tokens_by_req[req_idx] = [anchor]
                continue
            jobs.append(
                {
                    "req_idx": req_idx,
                    "req": req,
                    "anchor": anchor,
                    "remaining": remaining,
                    "base_len": req.get_total_length(),
                }
            )

        if not jobs:
            return output_tokens_by_req

        draft_budget = self.num_draft_tokens
        finite_remaining = [
            job["remaining"] - 1 for job in jobs if job["remaining"] is not None
        ]
        if finite_remaining:
            draft_budget = min(draft_budget, min(finite_remaining))
        if draft_budget <= 0:
            for job in jobs:
                output_tokens_by_req[job["req_idx"]] = [job["anchor"]]
            return output_tokens_by_req

        candidates = [
            {
                **job,
                # Reserve anchor + every drafted token before the draft writes
                # its masked query block into the independent draft KV pool.
                "draft_tokens": [0] * (draft_budget + 1),
            }
            for job in jobs
        ]
        fuse_target_hidden = (
            not scheduler_output.is_prefill
            and len(jobs) == len(requests)
            and draft_budget == self.num_draft_tokens
        )
        if not fuse_target_hidden:
            self._inject_target_hidden(model_input, target_output["hidden_states"])
        with cache_ops.verify_batch(candidates) as reservation:
            draft_input = self._build_draft_query_input(candidates, draft_budget)
            draft_input["draft_anchor_tokens"] = infinicore.from_list(
                [job["anchor"] for job in jobs], dtype=infinicore.int64
            )
            if fuse_target_hidden:
                draft_input["target_hidden_states"] = target_output["hidden_states"]
                draft_input["target_hidden_position_ids"] = model_input["position_ids"]
                draft_input["target_hidden_slot_mapping"] = model_input["slot_mapping"]
            draft_output = self.draft_model_engine.forward_raw(**draft_input)
            flat_draft_tokens = draft_output["output_ids"].to_numpy().tolist()
            if len(flat_draft_tokens) != len(jobs) * draft_budget:
                raise RuntimeError(
                    "DSpark proposal output count does not match draft width"
                )
            draft_tokens = [
                [
                    int(token)
                    for token in flat_draft_tokens[
                        row * draft_budget : (row + 1) * draft_budget
                    ]
                ]
                for row in range(len(jobs))
            ]
            for candidate, tokens in zip(candidates, draft_tokens):
                candidate["draft_tokens"] = [candidate["anchor"], *tokens]
                self.eagle_total_count += len(tokens)

            verify_input = self._build_paged_verify_batch_input(candidates)
            verify_input["capture_layer_ids"] = self.target_layer_ids
            reservation.mark_submitted()
            verify_output = self.target_model_engine.forward_raw(**verify_input)
            verify_token_ids = verify_output["output_ids"].to_numpy().tolist()
            self._inject_target_hidden(verify_input, verify_output["hidden_states"])
            reservation.mark_completed()

            expected_count = sum(len(row["draft_tokens"]) for row in candidates)
            if len(verify_token_ids) != expected_count:
                raise RuntimeError(
                    "DSpark verification output count does not match batch"
                )

            keep_lengths = []
            offset = 0
            for candidate in candidates:
                req = candidate["req"]
                proposed = candidate["draft_tokens"]
                segment = verify_token_ids[offset : offset + len(proposed)]
                offset += len(proposed)
                accepted = 1
                correction = None
                for draft_idx in range(1, len(proposed)):
                    expected = int(segment[draft_idx - 1])
                    if proposed[draft_idx] != expected:
                        correction = expected
                        break
                    accepted += 1
                if correction is None:
                    correction = int(segment[-1])

                output_tokens = proposed[:accepted] + [correction]
                remaining = candidate["remaining"]
                if remaining is not None:
                    output_tokens = output_tokens[:remaining]
                if not req.sampling_params.ignore_eos:
                    for index, token in enumerate(output_tokens):
                        if token in req.eos_token_ids:
                            output_tokens = output_tokens[: index + 1]
                            break
                accepted = min(accepted, len(output_tokens))
                self.eagle_accept_count += max(accepted - 1, 0)
                keep_lengths.append(candidate["base_len"] + accepted)
                output_tokens_by_req[candidate["req_idx"]] = output_tokens

            reservation.commit(keep_lengths)
        return output_tokens_by_req

    def _forward_device_token_cycle(
        self,
        scheduler_output,
        model_input,
        target_output,
        remaining_by_req,
        cache_ops,
    ):
        requests = scheduler_output.scheduled_requests
        batch_size = len(requests)
        anchors = target_output["output_ids"]
        if anchors.numel() != batch_size:
            raise RuntimeError(
                "DSpark target output count does not match request count"
            )

        jobs = [
            {
                "req_idx": req_idx,
                "req": req,
                "remaining": remaining,
                "base_len": req.get_total_length(),
            }
            for req_idx, (req, remaining) in enumerate(zip(requests, remaining_by_req))
        ]
        candidates = [
            {
                **job,
                "draft_tokens": [0] * self.verify_num_draft_tokens,
            }
            for job in jobs
        ]
        fuse_target_hidden = not scheduler_output.is_prefill
        if not fuse_target_hidden:
            self._inject_target_hidden(model_input, target_output["hidden_states"])

        with cache_ops.verify_batch(candidates) as reservation:
            draft_input = self._build_draft_query_input(
                candidates, self.num_draft_tokens
            )
            draft_input["draft_anchor_tokens"] = anchors
            draft_input["output_token_prefix"] = anchors
            draft_input["output_ids_on_device"] = True
            if fuse_target_hidden:
                draft_input["target_hidden_states"] = target_output["hidden_states"]
                draft_input["target_hidden_position_ids"] = model_input["position_ids"]
                draft_input["target_hidden_slot_mapping"] = model_input["slot_mapping"]
            draft_output = self.draft_model_engine.forward_raw(**draft_input)
            proposed_tokens = draft_output["output_ids"]
            expected_proposals = batch_size * self.verify_num_draft_tokens
            if proposed_tokens.numel() != expected_proposals:
                raise RuntimeError(
                    "DSpark proposal output count does not match draft width"
                )
            self.eagle_total_count += batch_size * self.num_draft_tokens

            verify_input = self._build_paged_verify_batch_input(candidates)
            verify_input["input_ids"] = proposed_tokens.view([1, expected_proposals])
            verify_input["capture_layer_ids"] = self.target_layer_ids
            verify_input["speculative_proposed_tokens"] = proposed_tokens
            reservation.mark_submitted()
            verify_output = self.target_model_engine.forward_raw(**verify_input)
            paired_tokens = (
                verify_output["output_ids"]
                .to_numpy()
                .reshape(batch_size, 2 * self.verify_num_draft_tokens)
            )
            reservation.mark_completed()

            output_tokens_by_req = [[] for _ in requests]
            keep_lengths = []
            width = self.verify_num_draft_tokens
            for candidate, row in zip(candidates, paired_tokens):
                req = candidate["req"]
                proposed = row[:width].tolist()
                segment = row[width:].tolist()
                accepted = 1
                correction = None
                for draft_idx in range(1, width):
                    expected = int(segment[draft_idx - 1])
                    if int(proposed[draft_idx]) != expected:
                        correction = expected
                        break
                    accepted += 1
                if correction is None:
                    correction = int(segment[-1])

                output_tokens = [
                    *[int(token) for token in proposed[:accepted]],
                    correction,
                ]
                remaining = candidate["remaining"]
                if remaining is not None:
                    output_tokens = output_tokens[:remaining]
                if not req.sampling_params.ignore_eos:
                    for index, token in enumerate(output_tokens):
                        if token in req.eos_token_ids:
                            output_tokens = output_tokens[: index + 1]
                            break
                accepted = min(accepted, len(output_tokens))
                self.eagle_accept_count += max(accepted - 1, 0)
                keep_lengths.append(candidate["base_len"] + accepted)
                output_tokens_by_req[candidate["req_idx"]] = output_tokens

            reservation.commit(keep_lengths)
            self._defer_or_inject_target_hidden(
                requests,
                verify_input,
                verify_output,
                output_tokens_by_req,
                remaining_by_req,
            )
        return output_tokens_by_req

    def _forward_stateful_device_token_cycle(
        self,
        scheduler_output,
        model_input,
        remaining_by_req,
        cache_ops,
    ):
        # The last generated token is the previous verification's correction
        # token. It is not in the target cache yet, but the draft cache already
        # contains every accepted predecessor state. Query the draft from that
        # token, then verify the token and its proposals in one target forward.
        requests = scheduler_output.scheduled_requests
        batch_size = len(requests)
        candidates = [
            {
                "req_idx": req_idx,
                "req": req,
                "remaining": remaining,
                "base_len": req.get_total_length(),
                "draft_tokens": [0] * self.num_draft_tokens,
            }
            for req_idx, (req, remaining) in enumerate(zip(requests, remaining_by_req))
        ]

        with cache_ops.verify_batch(candidates) as reservation:
            draft_input = self._build_stateful_draft_query_input(candidates)
            anchors = model_input["input_ids"].view([batch_size])
            draft_input["draft_anchor_tokens"] = anchors
            draft_input["output_token_prefix"] = anchors
            draft_input["output_ids_on_device"] = True
            self._attach_pending_target_hidden(requests, draft_input)
            draft_output = self.draft_model_engine.forward_raw(**draft_input)
            verify_tokens = draft_output["output_ids"]
            expected_tokens = batch_size * self.verify_num_draft_tokens
            if verify_tokens.numel() != expected_tokens:
                raise RuntimeError(
                    "DSpark stateful proposal output count does not match verify width"
                )
            self.eagle_total_count += batch_size * self.num_draft_tokens

            verify_input = self._build_stateful_verify_batch_input(candidates)
            verify_input["input_ids"] = verify_tokens.view([1, expected_tokens])
            verify_input["capture_layer_ids"] = self.target_layer_ids
            verify_input["speculative_proposed_tokens"] = verify_tokens
            reservation.mark_submitted()
            verify_output = self.target_model_engine.forward_raw(**verify_input)
            paired_tokens = (
                verify_output["output_ids"]
                .to_numpy()
                .reshape(batch_size, 2 * self.verify_num_draft_tokens)
            )
            reservation.mark_completed()

            output_tokens_by_req = [[] for _ in requests]
            keep_lengths = []
            width = self.verify_num_draft_tokens
            for candidate, row in zip(candidates, paired_tokens):
                req = candidate["req"]
                proposed = row[:width].tolist()
                segment = row[width:].tolist()
                accepted = 0
                correction = None
                for draft_idx in range(1, width):
                    expected = int(segment[draft_idx - 1])
                    if int(proposed[draft_idx]) != expected:
                        correction = expected
                        break
                    accepted += 1
                if correction is None:
                    correction = int(segment[-1])

                output_tokens = [
                    *[int(token) for token in proposed[1 : accepted + 1]],
                    correction,
                ]
                remaining = candidate["remaining"]
                if remaining is not None:
                    output_tokens = output_tokens[:remaining]
                if not req.sampling_params.ignore_eos:
                    for index, token in enumerate(output_tokens):
                        if token in req.eos_token_ids:
                            output_tokens = output_tokens[: index + 1]
                            break
                accepted = min(accepted, len(output_tokens))
                self.eagle_accept_count += accepted
                keep_lengths.append(candidate["base_len"] + accepted)
                output_tokens_by_req[candidate["req_idx"]] = output_tokens

            reservation.commit(keep_lengths)
            self._defer_or_inject_target_hidden(
                requests,
                verify_input,
                verify_output,
                output_tokens_by_req,
                remaining_by_req,
            )
        return output_tokens_by_req

    def _pending_matches(self, requests):
        pending = self._pending_target_hidden
        return (
            pending is not None
            and len(pending["requests"]) == len(requests)
            and all(
                previous is current
                for previous, current in zip(pending["requests"], requests)
            )
        )

    def _attach_pending_target_hidden(self, requests, draft_input):
        if not self._pending_matches(requests):
            return
        pending = self._pending_target_hidden
        self._pending_target_hidden = None
        source_input = pending["model_input"]
        draft_input["target_hidden_states"] = pending["hidden_states"]
        draft_input["target_hidden_position_ids"] = source_input["position_ids"]
        draft_input["target_hidden_slot_mapping"] = source_input["slot_mapping"]

    def _flush_pending_target_hidden(self):
        pending = self._pending_target_hidden
        if pending is None:
            return
        self._pending_target_hidden = None
        self._inject_target_hidden(
            pending["model_input"],
            pending["hidden_states"],
            dependency_event=None,
        )

    def _defer_or_inject_target_hidden(
        self,
        requests,
        verify_input,
        verify_output,
        output_tokens_by_req,
        remaining_by_req,
    ):
        can_defer = True
        for req, tokens, remaining in zip(
            requests, output_tokens_by_req, remaining_by_req
        ):
            if not req.sampling_params.ignore_eos and any(
                token in req.eos_token_ids for token in tokens
            ):
                can_defer = False
                break
            if (
                remaining is not None
                and remaining - len(tokens) < self.verify_num_draft_tokens
            ):
                can_defer = False
                break

        if not can_defer:
            self._inject_target_hidden(
                verify_input,
                verify_output["hidden_states"],
                dependency_event=None,
            )
            return

        self._pending_target_hidden = {
            "requests": tuple(requests),
            "model_input": verify_input,
            "hidden_states": verify_output["hidden_states"],
            "completion_event": verify_output["completion_event"],
        }

    def _inject_target_hidden(self, model_input, hidden_states, dependency_event=None):
        self.draft_model_engine.forward_raw(
            input_ids=model_input["input_ids"],
            position_ids=model_input["position_ids"],
            past_kv_lengths=model_input["past_kv_lengths"],
            total_kv_lengths=model_input["total_kv_lengths"],
            input_offsets=model_input["input_offsets"],
            cu_seqlens=model_input["cu_seqlens"],
            block_tables=model_input["block_tables"],
            slot_mapping=model_input["slot_mapping"],
            target_hidden_states=hidden_states,
            dependency_event=dependency_event,
            sample_all_positions=False,
            cache_only=True,
            skip_sampling=True,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )

    def _build_draft_query_input(self, candidates, draft_budget):
        tokens = []
        positions = []
        past_lens = []
        total_lens = []
        input_offsets = [0]
        cu_seqlens = [0]
        slot_mapping = []
        block_width = max(len(candidate["block_table"]) for candidate in candidates)
        block_tables = []
        for candidate in candidates:
            base = candidate["base_len"]
            anchor = candidate.get("anchor")
            tokens.extend(
                [self.mask_token_id] * draft_budget
                if anchor is None
                else [anchor, *([self.mask_token_id] * (draft_budget - 1))]
            )
            positions.extend(range(base, base + draft_budget))
            past_lens.append(base)
            total_lens.append(base + draft_budget)
            input_offsets.append(input_offsets[-1] + draft_budget)
            cu_seqlens.append(cu_seqlens[-1] + base + draft_budget)
            slot_mapping.extend(candidate["slot_mapping"][:draft_budget])
            block_tables.append(
                candidate["block_table"]
                + [-1] * (block_width - len(candidate["block_table"]))
            )
        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(positions, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(past_lens, dtype=infinicore.int32),
            "total_kv_lengths": infinicore.from_list(
                total_lens, dtype=infinicore.int32
            ),
            "input_offsets": infinicore.from_list(
                input_offsets, dtype=infinicore.int32
            ),
            "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_all_positions": True,
        }

    def _build_stateful_draft_query_input(self, candidates):
        tokens = []
        positions = []
        past_lens = []
        total_lens = []
        input_offsets = [0]
        cu_seqlens = [0]
        slot_mapping = []
        block_width = max(len(candidate["block_table"]) for candidate in candidates)
        for candidate in candidates:
            req = candidate["req"]
            base = candidate["base_len"]
            if len(req.slot_mapping) != 1:
                raise RuntimeError(
                    "DSpark stateful decode requires one current-token cache slot"
                )
            tokens.extend([self.mask_token_id] * self.num_draft_tokens)
            positions.extend(range(base - 1, base - 1 + self.num_draft_tokens))
            past_lens.append(base - 1)
            total_lens.append(base - 1 + self.num_draft_tokens)
            input_offsets.append(input_offsets[-1] + self.num_draft_tokens)
            cu_seqlens.append(cu_seqlens[-1] + total_lens[-1])
            slot_mapping.append(req.slot_mapping[0])
            slot_mapping.extend(candidate["slot_mapping"][: self.num_draft_tokens - 1])
            block_tables = candidate["block_table"] + [-1] * (
                block_width - len(candidate["block_table"])
            )
            candidate["stateful_block_table"] = block_tables

        return {
            "input_ids": self._pinned_from_list(
                "draft_input_ids", [tokens], infinicore.int64
            ),
            "position_ids": self._pinned_from_list(
                "draft_position_ids", positions, infinicore.int64
            ),
            "past_kv_lengths": self._pinned_from_list(
                "draft_past_kv_lengths", past_lens, infinicore.int32
            ),
            "total_kv_lengths": self._pinned_from_list(
                "draft_total_kv_lengths", total_lens, infinicore.int32
            ),
            "input_offsets": self._pinned_from_list(
                "draft_input_offsets", input_offsets, infinicore.int32
            ),
            "cu_seqlens": self._pinned_from_list(
                "draft_cu_seqlens", cu_seqlens, infinicore.int32
            ),
            "block_tables": self._pinned_from_list(
                "draft_block_tables",
                [candidate["stateful_block_table"] for candidate in candidates],
                infinicore.int32,
            ),
            "slot_mapping": self._pinned_from_list(
                "draft_slot_mapping", slot_mapping, infinicore.int64
            ),
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_all_positions": True,
        }

    def _build_stateful_verify_batch_input(self, candidates):
        metadata_slot = self._verify_metadata_slot
        self._verify_metadata_slot ^= 1
        positions = []
        past_lens = []
        total_lens = []
        input_offsets = [0]
        cu_seqlens = [0]
        slot_mapping = []
        for candidate in candidates:
            req = candidate["req"]
            base = candidate["base_len"]
            positions.extend(range(base - 1, base - 1 + self.verify_num_draft_tokens))
            past_lens.append(base - 1)
            total_lens.append(base - 1 + self.verify_num_draft_tokens)
            input_offsets.append(input_offsets[-1] + self.verify_num_draft_tokens)
            cu_seqlens.append(cu_seqlens[-1] + total_lens[-1])
            slot_mapping.append(req.slot_mapping[0])
            slot_mapping.extend(candidate["slot_mapping"])

        return {
            "position_ids": self._pinned_from_list(
                f"verify_position_ids_{metadata_slot}",
                positions,
                infinicore.int64,
            ),
            "past_kv_lengths": self._pinned_from_list(
                "verify_past_kv_lengths", past_lens, infinicore.int32
            ),
            "total_kv_lengths": self._pinned_from_list(
                "verify_total_kv_lengths", total_lens, infinicore.int32
            ),
            "input_offsets": self._pinned_from_list(
                "verify_input_offsets", input_offsets, infinicore.int32
            ),
            "cu_seqlens": self._pinned_from_list(
                "verify_cu_seqlens", cu_seqlens, infinicore.int32
            ),
            "block_tables": self._pinned_from_list(
                "verify_block_tables",
                [candidate["stateful_block_table"] for candidate in candidates],
                infinicore.int32,
            ),
            "slot_mapping": self._pinned_from_list(
                f"verify_slot_mapping_{metadata_slot}",
                slot_mapping,
                infinicore.int64,
            ),
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_all_positions": True,
        }

    def _pinned_from_list(self, name, values, dtype):
        shape = [len(values)]
        if values and isinstance(values[0], (list, tuple)):
            shape.append(len(values[0]))
        tensor = self._pinned_metadata.get(name)
        if tensor is None or tensor.shape != shape:
            tensor = infinicore.from_list(values, dtype=dtype, pin_memory=True)
            self._pinned_metadata[name] = tensor
        else:
            tensor.copy_from_list_(values)
        return tensor
