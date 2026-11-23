from ... import _core as infinicore


class DeepseekV4MtpRunner:
    """One-step speculative runner backed by the checkpoint's MTP layer."""

    def __init__(self, config, target_model_engine):
        if config.cache_type != "paged":
            raise ValueError("DeepSeek V4 MTP requires paged KV cache")
        if config.num_draft_tokens < 1:
            raise ValueError("DeepSeek V4 MTP requires num_draft_tokens >= 1")
        self.config = config
        self.target_model_engine = target_model_engine
        self.mtp_accept_count = 0
        self.mtp_total_count = 0

    def forward(self, scheduler_output, model_input):
        cache_ops = getattr(scheduler_output, "speculative_cache_ops", None)
        if cache_ops is None:
            return self.target_model_engine.forward(**model_input).to_numpy().tolist()

        requests = scheduler_output.scheduled_requests
        if not requests:
            return []
        if len(requests) != 1:
            raise ValueError("Deepseek V4 MTP currently requires max_batch_size=1")

        target_input = dict(model_input)
        target_input["sample_all_positions"] = scheduler_output.is_prefill
        target_output = self.target_model_engine.forward_raw(**target_input)
        target_token_ids = target_output["output_ids"].to_numpy().tolist()
        if not target_token_ids:
            return target_token_ids

        input_offsets = model_input["input_offsets"].to_numpy().tolist()
        target_tokens = [
            int(target_token_ids[int(input_offsets[index + 1]) - 1])
            if scheduler_output.is_prefill
            else int(target_token_ids[index])
            for index in range(len(requests))
        ]
        outputs: list[list[int]] = [[] for _ in requests]
        candidates = []
        for index, (request, target_token) in enumerate(
            zip(requests, target_tokens)
        ):
            max_tokens = request.sampling_params.max_tokens
            remaining = (
                None
                if max_tokens is None
                else max_tokens - request.get_num_generated_tokens()
            )
            if remaining is not None and remaining <= 1:
                outputs[index] = [target_token]
                continue
            candidates.append(
                {
                    "req_idx": index,
                    "req": request,
                    "base_len": request.get_total_length(),
                    "remaining": remaining,
                    "target_token": target_token,
                }
            )

        if not candidates:
            return outputs

        mtp_input = self._build_mtp_input(
            model_input, target_output["hidden_states"], target_tokens
        )
        mtp_output = self.target_model_engine.forward_raw(**mtp_input)
        mtp_tokens = mtp_output["output_ids"].to_numpy().tolist()
        if len(mtp_tokens) != len(requests):
            raise RuntimeError("DeepSeek V4 MTP output count does not match batch")

        verify_candidates = []
        for candidate in candidates:
            req_idx = candidate["req_idx"]
            draft_tokens = [
                candidate["target_token"],
                int(mtp_tokens[req_idx]),
            ]
            verify_candidates.append({**candidate, "draft_tokens": draft_tokens})
            self.mtp_total_count += 1

        with cache_ops.verify_batch(verify_candidates) as reservation:
            verify_input = self._build_paged_verify_batch_input(verify_candidates)
            reservation.mark_submitted()
            verify_output = self.target_model_engine.forward_raw(**verify_input)
            verify_token_ids = verify_output["output_ids"].to_numpy().tolist()
            expected_count = sum(
                len(row["draft_tokens"]) for row in verify_candidates
            )
            if len(verify_token_ids) != expected_count:
                raise RuntimeError(
                    "DeepSeek V4 MTP verification output count does not match batch"
                )
            if any(type(token) is not int or token < 0 for token in verify_token_ids):
                raise RuntimeError(
                    "DeepSeek V4 MTP verification requires integer token IDs"
                )

            verify_offsets = verify_input["input_offsets"].to_numpy().tolist()
            refresh_tokens = [
                int(verify_token_ids[int(verify_offsets[index + 1]) - 1])
                for index in range(len(verify_candidates))
            ]
            refresh_input = self._build_mtp_input(
                verify_input, verify_output["hidden_states"], refresh_tokens
            )
            self.target_model_engine.forward_raw(**refresh_input)
            reservation.mark_completed()

            keep_lengths = []
            offset = 0
            for candidate in verify_candidates:
                request = candidate["req"]
                draft_tokens = candidate["draft_tokens"]
                segment = verify_token_ids[offset : offset + len(draft_tokens)]
                offset += len(draft_tokens)

                accepted = 1
                if draft_tokens[1] == segment[0]:
                    accepted = 2
                    correction = int(segment[1])
                    self.mtp_accept_count += 1
                else:
                    correction = int(segment[0])
                output_tokens = draft_tokens[:accepted] + [correction]
                remaining = candidate["remaining"]
                if remaining is not None:
                    output_tokens = output_tokens[:remaining]
                if not request.sampling_params.ignore_eos:
                    for token_index, token in enumerate(output_tokens):
                        if token in request.eos_token_ids:
                            output_tokens = output_tokens[: token_index + 1]
                            break
                accepted = min(accepted, len(output_tokens))
                keep_lengths.append(candidate["base_len"] + accepted)
                outputs[candidate["req_idx"]] = output_tokens

            reservation.commit(keep_lengths)

        return outputs

    def _build_paged_verify_batch_input(self, candidates):
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
            "input_ids": infinicore.from_list_host(
                [tokens], dtype=infinicore.int64
            ),
            "position_ids": infinicore.from_list_host(
                position_ids, dtype=infinicore.int64
            ),
            "past_kv_lengths": infinicore.from_list_host(
                past_lens, dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list_host(
                seq_lens, dtype=infinicore.int32
            ),
            "input_offsets": infinicore.from_list_host(
                input_offsets, dtype=infinicore.int32
            ),
            "cu_seqlens": infinicore.from_list_host(
                cu_seqlens, dtype=infinicore.int32
            ),
            "block_tables": infinicore.from_list_host(
                block_tables, dtype=infinicore.int32
            ),
            "slot_mapping": infinicore.from_list_host(
                slot_mapping, dtype=infinicore.int64
            ),
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "sample_all_positions": True,
        }

    def _build_mtp_input(self, base_input, target_hidden_states, next_tokens):
        input_offsets = base_input["input_offsets"].to_numpy().tolist()
        flat_tokens = base_input["input_ids"].to_numpy().reshape(-1).tolist()
        if len(next_tokens) != len(input_offsets) - 1:
            raise ValueError("DeepSeek V4 MTP next-token batch mismatch")

        shifted_tokens = []
        for index, next_token in enumerate(next_tokens):
            start = int(input_offsets[index])
            end = int(input_offsets[index + 1])
            if end <= start:
                raise ValueError("DeepSeek V4 MTP requires non-empty query rows")
            shifted_tokens.extend(int(token) for token in flat_tokens[start + 1 : end])
            shifted_tokens.append(int(next_token))

        mtp_input = dict(base_input)
        mtp_input["input_ids"] = infinicore.from_list_host(
            [shifted_tokens], dtype=infinicore.int64
        )
        mtp_input["target_hidden_states"] = target_hidden_states
        mtp_input["temperature"] = 1.0
        mtp_input["top_k"] = 1
        mtp_input["top_p"] = 1.0
        mtp_input["sample_all_positions"] = False
        mtp_input.pop("sampling_rows", None)
        mtp_input.pop("random_values", None)
        return mtp_input
