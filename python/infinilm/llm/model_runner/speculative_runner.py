import logging

import infinicore
from infinilm.cache.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.draft_spec import (
    UnsupportedDraftError,
    draft_position_ids,
    explain_missing_draft,
    get_draft_model_spec,
    resolve_draft,
)
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file

logger = logging.getLogger(__name__)


def resolve_draft_engine_path(draft_model_path):
    """Directory the draft engine should be built from.

    A checkpoint that embeds its draft weights under a family key prefix needs
    a standalone draft config, which `infinilm.draft_spec` derives from the
    checkpoint's own metadata; other drafts are used as given.
    """
    checkpoint = resolve_draft(draft_model_path)
    return draft_model_path if checkpoint is None else checkpoint.engine_path


class SpeculativeRunner:
    def __init__(self, config, target_model_engine, device):
        self.config = config
        self.target_model_engine = target_model_engine
        self.num_draft_tokens = config.num_draft_tokens
        self.draft_max_batch_size = config.max_batch_size
        self.eagle_accept_count = 0
        self.eagle_total_count = 0
        # Rounds that fell back to the frozen single-token path because no
        # temporary state row was available for the verified batch.
        self.verify_scratch_exhausted = 0
        self.accepted_count_histogram: dict[int, int] = {}
        self._mamba_cache = None
        # Verification slots only exist on the paged cache, whose config is the
        # one that carries a block size. With a static cache the scheduler
        # hands out no speculative cache ops, so every request stays on the
        # plain target path and the drafting state is never used.
        target_block_size = getattr(
            target_model_engine.get_cache_config(), "block_size", None
        )
        self._cache_block_size = (
            target_block_size() if callable(target_block_size) else None
        )
        if self._cache_block_size is None:
            logger.warning(
                "Speculative decoding needs the paged KV cache: with a static "
                "cache the requests run non-speculatively and --draft-model "
                "has no effect."
            )

        draft_checkpoint = resolve_draft(config.draft_model_path)
        draft_model_path = resolve_draft_engine_path(config.draft_model_path)
        draft_cache_config = StaticKVCacheConfig(
            max_batch_size=config.max_batch_size, max_cache_len=config.max_cache_len
        )
        self.draft_model_engine = InferEngine(
            model_path=draft_model_path,
            device=device,
            distributed_config=DistConfig(config.tensor_parallel_size),
            cache_config=draft_cache_config,
            enable_graph_compiling=config.enable_graph,
            attention_backend="default",
            use_mla=False,
            weight_load_mode=config.weight_load_mode,
        )
        self.draft_model_type = self.draft_model_engine.model_type
        if draft_checkpoint is not None:
            # The description that resolved the checkpoint is the one that
            # describes the engine the fixture was built for.
            self.draft_spec = draft_checkpoint.spec
            if self.draft_model_type != self.draft_spec.draft_model_type:
                raise UnsupportedDraftError(
                    f"the draft fixture declares model type "
                    f"{self.draft_spec.draft_model_type!r} but the engine built "
                    f"{self.draft_model_type!r}"
                )
        else:
            self.draft_spec = get_draft_model_spec(self.draft_model_type)
        if self.draft_spec is None:
            raise UnsupportedDraftError(
                f"--draft-model {config.draft_model_path} is not a supported "
                f"draft: {explain_missing_draft(config.draft_model_path, self.draft_model_type)}"
            )
        self._check_draft_against_target()
        if not config.skip_load:
            load_model_state_dict_by_file(
                self.draft_model_engine,
                draft_model_path,
                dtype=self.draft_model_engine.dtype,
            )

    def _check_draft_against_target(self):
        """Reject a draft whose vocabulary differs from the target's.

        A draft proposes tokens that the target then verifies, so both must
        index the same vocabulary; equal sizes are necessary but not sufficient
        and are the strongest statement the configs support.
        """
        target_config = self.target_model_engine.hf_config
        draft_config = self.draft_model_engine.hf_config
        target_vocab = target_config.get("text_config", target_config).get("vocab_size")
        draft_vocab = draft_config.get("text_config", draft_config).get("vocab_size")
        if target_vocab is not None and draft_vocab is not None:
            if target_vocab != draft_vocab:
                raise UnsupportedDraftError(
                    f"the draft model has vocab_size={draft_vocab} while the "
                    f"target has vocab_size={target_vocab}; a draft must share "
                    "the target's vocabulary (MODELS.md, criterion C5)"
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

        mamba_cache = cache_ops.mamba_cache()
        # Used by the post-verification state restore to release the rows.
        self._mamba_cache = mamba_cache

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

        # Verifying more than one token needs a temporary state row per request,
        # because the committed row must only advance by the accepted count.
        # Rows are counted here, before any draft runs: a request that cannot get
        # one drafts a single token instead, which is the frozen path that
        # advances the committed row in place and stays consistent.
        if mamba_cache is not None:
            free_rows = mamba_cache.get_num_free_blocks()
            needs_row = [job for job in draft_jobs if job["num_tokens"] > 1]
            # free_rows is both the number of rows available and the index of the
            # first request that cannot have one: rows are borrowed only after
            # this loop, so the count cannot go stale between the two uses. A
            # reservation moved after a borrow would make the slice wrong.
            for job in needs_row[free_rows:]:
                self.verify_scratch_exhausted += 1
                logger.warning(
                    "No free mamba state row for request %s; drafting a single "
                    "token this round (free rows: %d)",
                    job["req"].request_id,
                    free_rows,
                )
                job["num_tokens"] = 1

        draft_results = self._draft_eagle_tokens_batch(draft_jobs)
        verify_candidates = []
        state_replays = []
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
            verify_block_table, verify_slots = cache_ops.append_verify_slots(
                list(req.block_table),
                base_len + 1,
                len(draft_tokens),
            )
            req.block_table = verify_block_table
            req.num_blocks = len(req.block_table)
            # A multi-token verification must not advance the committed state:
            # it writes a temporary row while reading the committed one.
            scratch_index = self._borrow_state_scratch(
                mamba_cache, req, len(draft_tokens)
            )
            verify_candidates.append(
                {
                    "req_idx": req_idx,
                    "req": req,
                    "base_len": base_len,
                    "remaining": job["remaining"],
                    "draft_tokens": draft_tokens,
                    "slot_mapping": verify_slots,
                    "scratch_index": scratch_index,
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
                self.accepted_count_histogram[accepted] = (
                    self.accepted_count_histogram.get(accepted, 0) + 1
                )
                keep_tokens = candidate["base_len"] + accepted
                req.block_table = cache_ops.rollback_to_length(
                    req.block_table, keep_tokens
                )
                req.num_blocks = len(req.block_table)
                req.slot_mapping = []

                output_tokens = draft_tokens[:accepted] + [correction]
                remaining = candidate["remaining"]
                if remaining is not None:
                    output_tokens = output_tokens[:remaining]
                output_tokens_by_req[req_idx] = output_tokens

                if candidate["scratch_index"] is None:
                    continue
                if accepted == len(draft_tokens):
                    # Fully accepted: the temporary row already holds the kept
                    # sequence's state, so ownership just moves to it.
                    mamba_cache.swap_slots(
                        req.mamba_cache_index, candidate["scratch_index"]
                    )
                    req.mamba_cache_index = candidate["scratch_index"]
                else:
                    state_replays.append(
                        {
                            "req": req,
                            "base_len": candidate["base_len"],
                            "draft_tokens": draft_tokens,
                            "accepted": accepted,
                            "scratch_index": candidate["scratch_index"],
                        }
                    )

            self._restore_verified_states(state_replays)

        return output_tokens_by_req

    def _borrow_state_scratch(self, mamba_cache, req, num_draft_tokens):
        """Take a temporary row for a multi-token verification, else None.

        A single-token verification keeps the frozen path, which advances the
        committed row in place because its state already matches the kept
        sequence; models without state rows have nothing to borrow. Rows were
        already counted against the free pool before the draft ran, so every
        multi-token request is guaranteed to get one here.
        """
        if mamba_cache is None or req.mamba_cache_index is None:
            return None
        if num_draft_tokens < 2:
            return None
        scratch_index = mamba_cache.borrow_slot()
        if scratch_index is None:
            raise RuntimeError(
                f"state row for request {req.request_id} was not reserved"
            )
        return scratch_index

    def _restore_verified_states(self, state_replays: list[dict]) -> None:
        """Restore partially accepted requests to their accepted prefix.

        The committed row still holds the pre-verification state, so replaying
        the accepted tokens through the decode path advances it by exactly the
        accepted number of steps. All requests replay in one batched call.
        """
        if not state_replays:
            return
        if self._cache_block_size is None:
            raise RuntimeError(
                "state replay needs the paged cache block size, which the "
                "configured cache does not expose"
            )
        self.target_model_engine.forward_raw(
            **self._build_state_replay_batch_input(state_replays)
        )
        for replay in state_replays:
            # The replay wrote the committed row, so the request keeps its index;
            # the temporary row that ran ahead of it is released untouched.
            self._mamba_cache.release_slot(replay["scratch_index"])

    def _build_state_replay_batch_input(self, state_replays: list[dict]) -> dict:
        """Build a decode-shaped batch over the accepted tokens.

        Each request packs the tokens the verification already accepted as one
        multi-token request, so the decode path advances its committed row.

        Two position-id layouts reach the target: a packed batch like this one
        passes one flat position per token (`[N]`), which is what the paged
        kernels read, while the draft's serial step passes one position per
        batch row (`[B, 1]`) through `draft_position_ids`. The layout is per
        call, not per model, so a new batch builder has to say which one it
        produces rather than assume.
        """
        block_size = self._cache_block_size
        tokens = []
        position_ids = []
        past_lens = []
        seq_lens = []
        input_offsets = [0]
        cu_seqlens = [0]
        slot_mapping = []
        block_tables = []
        state_indices = []
        max_block_table_len = max(
            len(replay["req"].block_table) for replay in state_replays
        )

        for replay in state_replays:
            req = replay["req"]
            base_len = replay["base_len"]
            replay_tokens = replay["draft_tokens"][: replay["accepted"]]
            tokens.extend(replay_tokens)
            position_ids.extend(range(base_len, base_len + len(replay_tokens)))
            past_lens.append(base_len)
            seq_lens.append(base_len + len(replay_tokens))
            input_offsets.append(input_offsets[-1] + len(replay_tokens))
            cu_seqlens.append(cu_seqlens[-1] + base_len + len(replay_tokens))
            # Every paged batch writes its key/value entries at these slots.
            for token_idx in range(base_len, base_len + len(replay_tokens)):
                block_idx, block_offset = divmod(token_idx, block_size)
                slot_mapping.append(
                    req.block_table[block_idx] * block_size + block_offset
                )
            block_tables.append(
                req.block_table + [-1] * (max_block_table_len - len(req.block_table))
            )
            state_indices.append(req.mamba_cache_index)

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
            # The replay advances each request's committed row in place.
            "mamba_init_state_indices": infinicore.from_list(
                state_indices, dtype=infinicore.int32
            ),
            "mamba_final_state_indices": infinicore.from_list(
                state_indices, dtype=infinicore.int32
            ),
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        }

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
                position_ids=self._build_draft_position_ids(positions),
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

    def _build_draft_position_ids(self, positions: list[int]) -> infinicore.Tensor:
        return draft_position_ids(self.draft_spec, positions)

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

        verify_input = {
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

        # Hybrid targets carry recurrent linear-attention state per request.
        # A single-token verification advances each request's own slot in place,
        # because its state already matches the kept sequence. A multi-token
        # verification reads that slot but writes a temporary row, so a partial
        # acceptance is undone by restoring the committed row.
        state_indices = [candidate["req"].mamba_cache_index for candidate in candidates]
        if all(index is not None for index in state_indices):
            verify_input["mamba_init_state_indices"] = infinicore.from_list(
                state_indices, dtype=infinicore.int32
            )
            verify_input["mamba_final_state_indices"] = infinicore.from_list(
                [
                    candidate["scratch_index"]
                    if candidate["scratch_index"] is not None
                    else candidate["req"].mamba_cache_index
                    for candidate in candidates
                ],
                dtype=infinicore.int32,
            )
        # State the batch shape explicitly instead of letting the model infer it
        # from the packed layout. The shape is whether the batch holds several
        # tokens for any request, so a batch of single-token requests states the
        # decode shape and takes the same path a plain decode step takes.
        verify_input["mamba_multi_token_batch"] = any(
            len(candidate["draft_tokens"]) > 1 for candidate in candidates
        )
        return verify_input
