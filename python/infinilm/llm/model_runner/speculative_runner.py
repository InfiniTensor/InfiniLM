import infinicore
from infinilm.cache.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


class SpeculativeRunner:
    def __init__(self, config, target_model_engine, device):
        self.config = config
        self.target_model_engine = target_model_engine
        self.num_draft_tokens = config.num_draft_tokens
        self.draft_max_batch_size = config.max_batch_size
        self.eagle_accept_count = 0
        self.eagle_total_count = 0

        draft_cache_config = StaticKVCacheConfig(
            max_batch_size=config.max_batch_size, max_cache_len=config.max_cache_len
        )
        self.draft_model_engine = InferEngine(
            model_path=config.draft_model_path,
            device=device,
            distributed_config=DistConfig(config.tensor_parallel_size),
            cache_config=draft_cache_config,
            enable_graph_compiling=config.enable_graph,
            attention_backend="default",
            use_mla=False,
            weight_load_mode=config.weight_load_mode,
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

        # Keep non-greedy sampling on the established target path. Correct stochastic
        # speculative sampling needs distribution-level acceptance, while current MTP
        # verification is exact for greedy decoding.
        if self.config.top_k != 1 or self.config.temperature != 1.0:
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        requests = scheduler_output.scheduled_requests
        if not requests:
            return []

        target_output = self.target_model_engine.forward_raw(**model_input)
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

            base_len = req.get_total_length()
            total_token_ids = req.get_all_token_ids() + draft_tokens
            verify_block_table, verify_slots = cache_ops.append_verify_slots(
                list(req.block_table),
                base_len + 1,
                len(draft_tokens),
                total_token_ids,
            )
            req.block_table = verify_block_table
            req.num_blocks = len(req.block_table)
            verify_candidates.append(
                {
                    "req_idx": req_idx,
                    "req": req,
                    "base_len": base_len,
                    "remaining": job["remaining"],
                    "draft_tokens": draft_tokens,
                    "slot_mapping": verify_slots,
                }
            )

        if verify_candidates:
            verify_output = self.target_model_engine.forward_raw(
                **self._build_paged_verify_batch_input(verify_candidates)
            )
            verify_token_ids = verify_output["output_ids"].to_numpy().tolist()
            verify_offsets = [0]
            for candidate in verify_candidates:
                verify_offsets.append(
                    verify_offsets[-1] + len(candidate["draft_tokens"])
                )

            for idx, candidate in enumerate(verify_candidates):
                req = candidate["req"]
                req_idx = candidate["req_idx"]
                draft_tokens = candidate["draft_tokens"]
                segment = verify_token_ids[
                    verify_offsets[idx] : verify_offsets[idx + 1]
                ]
                accepted = 1
                correction = None
                for draft_idx in range(1, len(draft_tokens)):
                    expected = int(segment[draft_idx - 1])
                    if draft_tokens[draft_idx] != expected:
                        correction = expected
                        break
                    accepted += 1

                if correction is None:
                    correction = int(segment[len(draft_tokens) - 1])

                self.eagle_accept_count += accepted
                keep_tokens = candidate["base_len"] + accepted
                req.block_table = cache_ops.rollback_to_length(
                    req.block_table, keep_tokens
                )
                req.num_blocks = len(req.block_table)
                req.slot_mapping = []
                accepted_token_ids = req.get_all_token_ids() + draft_tokens[:accepted]
                cache_ops.commit_accepted_tokens(
                    req.block_table, accepted_token_ids, keep_tokens
                )

                output_tokens = draft_tokens[:accepted] + [correction]
                remaining = candidate["remaining"]
                if remaining is not None:
                    output_tokens = output_tokens[:remaining]
                output_tokens_by_req[req_idx] = output_tokens

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
            len(candidate["req"].block_table) for candidate in candidates
        )

        for candidate in candidates:
            req = candidate["req"]
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
                req.block_table + [-1] * (max_block_table_len - len(req.block_table))
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
        }
