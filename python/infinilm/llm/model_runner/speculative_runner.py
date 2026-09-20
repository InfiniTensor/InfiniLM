import logging
import os

import infinicore
from infinilm.cache.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file

logger = logging.getLogger(__name__)


class SpeculativeRunner:
    """Speculative decoding runner: draft → paged verify → accept/rollback.

    两种 draft 来源（config.speculative_method）：
    - "eagle": MiniCPM Eagle draft 头逐步自回归猜测（需要 draft_model_path，
      依赖 target 前向的 hidden states，主前向只能走 eager 的 forward_raw）。
    - "prompt_lookup": 零训练的 n-gram 后缀匹配，从请求自身的
      prompt+已生成序列里复制后续片段作为 draft。不依赖 hidden states；
      纯 decode 批次走融合单前向（主采样 + 验证一次完成，形状固定为
      b × (k+1)），混排/prefill 批次走两前向流程。

    两种方法共用同一条精确验证链路：append_verify_slots 申请临时 KV 槽 →
    全位置采样前向（融合或独立 verify）→ 接受最长匹配前缀 + 1 个修正
    token → rollback_to_length 回收未接受的槽位。贪心解码下输出与非投机
    数学等价（分布无损）；注意不保证逐位一致——verify 前向的 batch 形状
    与基线 decode 不同，logit 近平局的位置 argmax 可能翻转（5090 实测
    分歧点 top-2 间隙 0~0.125，见 dev_perf/gap_analysis.md v16）。大
    batch（> INFINILM_SPEC_MAX_BATCH_SIZE，默认 32）回退常规前向。

    自适应收益门控：滑动窗口（默认 32 个请求步）内平均每步产出低于
    INFINILM_SPEC_MIN_AVG_TOKENS（默认 2.0，eager 投机步成本 ≈ 2× 常规
    decode 步的盈亏平衡点）时，回退常规前向 INFINILM_SPEC_GATE_COOLDOWN
    步（默认 64）再自动重试——开放生成等低命中负载开投机不致亏。
    """

    def __init__(self, config, target_model_engine, device):
        self.config = config
        self.target_model_engine = target_model_engine
        self.speculative_method = config.speculative_method
        self.num_draft_tokens = config.num_draft_tokens
        self.draft_max_batch_size = config.max_batch_size
        # 接受率埋点：drafted/accepted 衡量 draft 质量，emitted_tokens/steps
        # 衡量端到端收益（非投机基线恒为 1.0 token/step）。
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_emitted_tokens = 0
        self.total_emitted_steps = 0
        self.total_verify_alloc_failures = 0

        # 大 batch 下 decode 转向计算受限，投机把每步计算量放大 k 倍反而
        # 降吞吐：batch 超过阈值时回退常规前向（env 可调）
        self.spec_max_batch_size = int(
            os.getenv("INFINILM_SPEC_MAX_BATCH_SIZE", "32")
        )

        # 自适应收益门控状态（窗口/冷静期均可 env 调，见类 docstring）
        self._gate_min_avg = float(
            os.getenv("INFINILM_SPEC_MIN_AVG_TOKENS", "2.0")
        )
        self._gate_window_size = int(os.getenv("INFINILM_SPEC_GATE_WINDOW", "32"))
        if self._gate_window_size < 1:
            raise ValueError("INFINILM_SPEC_GATE_WINDOW must be >= 1")
        self._gate_cooldown_len = int(
            os.getenv("INFINILM_SPEC_GATE_COOLDOWN", "64")
        )
        self._gate_window_tokens = 0
        self._gate_window_steps = 0
        self._gate_cooldown = 0
        self.gate_triggered_count = 0

        self.draft_model_engine = None
        if self.speculative_method == "prompt_lookup":
            # n-gram 后缀匹配的长度范围（仅影响命中质量，可调参不影响正确性）
            self.prompt_lookup_max_ngram = int(
                os.getenv("INFINILM_PROMPT_LOOKUP_MAX_NGRAM", "4")
            )
            self.prompt_lookup_min_ngram = int(
                os.getenv("INFINILM_PROMPT_LOOKUP_MIN_NGRAM", "2")
            )
            if (
                self.prompt_lookup_min_ngram < 1
                or self.prompt_lookup_max_ngram < self.prompt_lookup_min_ngram
            ):
                raise ValueError(
                    "prompt-lookup n-gram bounds must satisfy "
                    "1 <= INFINILM_PROMPT_LOOKUP_MIN_NGRAM "
                    "<= INFINILM_PROMPT_LOOKUP_MAX_NGRAM"
                )
            return

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

    def get_acceptance_stats(self) -> dict:
        """接受率/收益统计快照（计数自引擎启动起累计）。"""
        return {
            "method": self.speculative_method,
            "num_draft_tokens": self.num_draft_tokens,
            "drafted_tokens": self.total_drafted,
            "accepted_tokens": self.total_accepted,
            "accept_rate": (
                self.total_accepted / self.total_drafted
                if self.total_drafted
                else None
            ),
            "spec_steps": self.total_emitted_steps,
            "emitted_tokens": self.total_emitted_tokens,
            "avg_tokens_per_step": (
                self.total_emitted_tokens / self.total_emitted_steps
                if self.total_emitted_steps
                else None
            ),
            "verify_alloc_failures": self.total_verify_alloc_failures,
            "gate_triggered": self.gate_triggered_count,
        }

    def forward(self, scheduler_output, model_input):
        cache_ops = getattr(scheduler_output, "speculative_cache_ops", None)
        if cache_ops is None:
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        # Keep non-greedy sampling on the established target path. Correct stochastic
        # speculative sampling needs distribution-level acceptance, while current
        # verification is exact for greedy decoding.
        if self.config.top_k != 1 or self.config.temperature != 1.0:
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        requests = scheduler_output.scheduled_requests
        if not requests:
            return []

        # 大 batch 下 decode 转向计算受限，投机把每步计算量放大 k 倍反而
        # 降吞吐：超过阈值直接走常规前向（decode 形状仍可命中 CUDA graph）
        if len(requests) > self.spec_max_batch_size:
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        # 收益门控冷静期：回退常规前向（decode 形状仍可命中 CUDA graph），
        # 冷静期结束后自动重试投机
        if self._gate_cooldown > 0:
            self._gate_cooldown -= 1
            sampled_tokens = self.target_model_engine.forward(**model_input)
            return sampled_tokens.to_numpy().tolist()

        use_prompt_lookup = self.speculative_method == "prompt_lookup"
        # 纯 decode 批次的 prompt_lookup 走融合单前向：一次前向同时完成
        # 主采样与 draft 验证，省掉独立 verify 前向的 launch/CPU 开销。
        # 含 prefill chunk 的混排批次仍走两前向流程（draft 在主前向之后）。
        if use_prompt_lookup and self._is_pure_decode_batch(
            requests, scheduler_output
        ):
            return self._forward_fused_prompt_lookup(
                scheduler_output, cache_ops, requests
            )

        if use_prompt_lookup:
            # 主前向只取每请求最后位置的采样结果
            sampled = self.target_model_engine.forward(**model_input)
            target_token_ids = sampled.to_numpy().tolist()
            hidden_states = None
        else:
            target_output = self.target_model_engine.forward_raw(**model_input)
            target_token_ids = target_output["output_ids"].to_numpy().tolist()
            hidden_states = target_output["hidden_states"]
        if not target_token_ids:
            return target_token_ids

        input_offsets = model_input["input_offsets"].to_numpy().tolist()
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        output_tokens_by_req: list[list[int]] = [[] for _ in requests]
        draft_jobs = []

        for req_idx, req in enumerate(requests):
            # 逐请求阶段判定（chunked prefill 混排批次中 batch 级 is_prefill
            # 已退化为"是否包含 chunk"）：prompt 未算完的中段 chunk 请求本步
            # 不产生 token（废 token 由 llm.py 的 _update_requests 丢弃），
            # 也绝不能进入 draft/verify——其序列末端不是真实生成位置，
            # 追加 verify 槽位会破坏块表不变量。
            if num_scheduled is not None:
                chunk_end = req.num_local_cached_tokens + num_scheduled[req_idx]
                if chunk_end < req.get_prompt_length():
                    output_tokens_by_req[req_idx] = []
                    continue
                is_prefill_req = (
                    req.num_local_cached_tokens < req.get_prompt_length()
                )
            else:
                is_prefill_req = scheduler_output.is_prefill

            if use_prompt_lookup:
                target_token = int(target_token_ids[req_idx])
            else:
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

            job = {
                "req_idx": req_idx,
                "req": req,
                "target_token": target_token,
                "remaining": remaining,
                "num_tokens": draft_budget,
            }
            if not use_prompt_lookup:
                (
                    source_token,
                    source_position,
                ) = self._get_last_input_token_and_position(req, is_prefill_req)
                job["source_token"] = source_token
                job["source_position"] = source_position
                job["target_hidden"] = hidden_states.narrow(1, last_input_idx, 1)
            draft_jobs.append(job)

        if use_prompt_lookup:
            draft_results = [
                self._draft_prompt_lookup_tokens(job["req"], job["num_tokens"])
                for job in draft_jobs
            ]
        else:
            draft_results = self._draft_eagle_tokens_batch(draft_jobs)

        verify_candidates = []
        for job, draft_tokens in zip(draft_jobs, draft_results):
            req_idx = job["req_idx"]
            req = job["req"]
            target_token = job["target_token"]
            if not draft_tokens:
                output_tokens_by_req[req_idx] = [target_token]
                continue

            self.total_drafted += len(draft_tokens)
            if draft_tokens[0] != target_token:
                output_tokens_by_req[req_idx] = [target_token]
                continue

            base_len = req.get_total_length()
            try:
                verify_block_table, verify_slots = cache_ops.append_verify_slots(
                    list(req.block_table),
                    base_len + 1,
                    len(draft_tokens),
                )
            except RuntimeError:
                # 临时验证槽位分配失败（KV 块不足）：退化为只产出 target
                # token，等价于非投机的一步，不阻塞正常 decode。
                self.total_verify_alloc_failures += 1
                output_tokens_by_req[req_idx] = [target_token]
                continue
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

                self.total_accepted += accepted
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

        self._record_step_stats(output_tokens_by_req)

        return output_tokens_by_req

    def _record_step_stats(self, output_tokens_by_req):
        """累积端到端收益统计，并维护自适应门控的滑动窗口。

        窗口攒满后结算一次：平均每步产出低于盈亏平衡阈值则进入冷静期
        （forward 开头回退常规前向），窗口清零、冷静期满后自动重试。
        """
        for output_tokens in output_tokens_by_req:
            if output_tokens:
                self.total_emitted_steps += 1
                self.total_emitted_tokens += len(output_tokens)
                self._gate_window_steps += 1
                self._gate_window_tokens += len(output_tokens)
        if self._gate_window_steps >= self._gate_window_size:
            avg = self._gate_window_tokens / self._gate_window_steps
            self._gate_window_steps = 0
            self._gate_window_tokens = 0
            if avg < self._gate_min_avg:
                self._gate_cooldown = self._gate_cooldown_len
                self.gate_triggered_count += 1
                logger.info(
                    "spec decoding gated: avg %.2f tok/step < %.2f over the "
                    "last window, fallback to plain decode for %d steps",
                    avg,
                    self._gate_min_avg,
                    self._gate_cooldown_len,
                )

    def _is_pure_decode_batch(self, requests, scheduler_output) -> bool:
        """批次内所有请求都处于 decode 相位（本步调度的 token 位于序列尾部）。

        chunked 混排中的中段/完成 chunk（num_local_cached_tokens <
        prompt_length）与旧式整段 prefill 批次都不算纯 decode。
        """
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if num_scheduled is None:
            return not scheduler_output.is_prefill
        return all(
            req.num_local_cached_tokens >= req.get_prompt_length()
            for req in requests
        )

    def _forward_fused_prompt_lookup(
        self, scheduler_output, cache_ops, requests
    ) -> list[list[int]]:
        """prompt_lookup 专用融合前向：主采样 + draft 验证一次完成。

        每个 decode 请求贡献固定 k+1 个 token 的一行：[已提交尾 token x,
        draft_1..draft_k]（draft 不足 k 个时用尾 token 补齐——验证逻辑对
        任意 draft 内容自校正，pad 最多造成巧合接受，不影响正确性）。
        批次形状固定为 b × (k+1)，是后续 verify 图化要录制的形状。

        输出与两前向流程逐 token 等价：行输出 o[0] 即 target_token（位置
        T 的预测）；o[j] 是 draft_j 之后位置的预测。接受 drafts 的最长
        匹配前缀 + 1 个修正 token，未接受的 KV 槽位回滚。
        """
        k = self.num_draft_tokens
        candidates = []
        for req_idx, req in enumerate(requests):
            base_len = req.get_total_length()
            max_tokens = req.sampling_params.max_tokens
            remaining = (
                None
                if max_tokens is None
                else max_tokens - req.get_num_generated_tokens()
            )
            last_token = (
                req.generated_token_ids[-1]
                if req.generated_token_ids
                else req.prompt_token_ids[-1]
            )
            real_drafts = (
                self._draft_prompt_lookup_tokens(req, k)
                if remaining is None or remaining > 1
                else []
            )
            drafts = list(real_drafts) + [int(last_token)] * (
                k - len(real_drafts)
            )

            # remaining<=1 时与两前向路径一样短路：本步只跑尾 token
            # （1 token 行），不分配 verify 槽位——既不浪费 k+1 行前向，
            # 也不会把本不需要的分配计进 verify_alloc_failures。
            if remaining is not None and remaining <= 1:
                verify_slots = None
            else:
                # x（位置 base_len-1）的槽位已由调度器分配；draft 占用
                # base_len..base_len+k-1 的临时槽位，验证后按接受长度回滚
                try:
                    block_table, verify_slots = cache_ops.append_verify_slots(
                        list(req.block_table), base_len + 1, k
                    )
                except RuntimeError:
                    # KV 块不足：本请求只跑尾 token（1 token 行），不投机
                    self.total_verify_alloc_failures += 1
                    verify_slots = None
                else:
                    req.block_table = block_table
                    req.num_blocks = len(req.block_table)

            candidates.append(
                {
                    "req_idx": req_idx,
                    "req": req,
                    "base_len": base_len,
                    "remaining": remaining,
                    "last_token": int(last_token),
                    "real_draft_len": len(real_drafts),
                    "drafts": drafts,
                    "verify_slots": verify_slots,
                }
            )

        tokens = []
        position_ids = []
        past_lens = []
        seq_lens = []
        input_offsets = [0]
        cu_seqlens = [0]
        slot_mapping = []
        block_tables = []
        max_block_table_len = max(len(c["req"].block_table) for c in candidates)

        for c in candidates:
            req = c["req"]
            base_len = c["base_len"]
            row = (
                [c["last_token"]] + c["drafts"]
                if c["verify_slots"] is not None
                else [c["last_token"]]
            )
            tokens.extend(row)
            position_ids.extend(range(base_len - 1, base_len - 1 + len(row)))
            past_lens.append(base_len - 1)
            seq_lens.append(base_len - 1 + len(row))
            input_offsets.append(input_offsets[-1] + len(row))
            cu_seqlens.append(cu_seqlens[-1] + base_len - 1 + len(row))
            slot_mapping.extend(req.slot_mapping)
            if c["verify_slots"] is not None:
                slot_mapping.extend(c["verify_slots"])
            block_tables.append(
                req.block_table
                + [-1] * (max_block_table_len - len(req.block_table))
            )

        fused_output = self.target_model_engine.forward_raw(
            input_ids=infinicore.from_list([tokens], dtype=infinicore.int64),
            position_ids=infinicore.from_list(position_ids, dtype=infinicore.int64),
            past_kv_lengths=infinicore.from_list(past_lens, dtype=infinicore.int32),
            total_kv_lengths=infinicore.from_list(seq_lens, dtype=infinicore.int32),
            input_offsets=infinicore.from_list(input_offsets, dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            block_tables=infinicore.from_list(block_tables, dtype=infinicore.int32),
            slot_mapping=infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        )
        out_ids = fused_output["output_ids"].to_numpy().tolist()

        output_tokens_by_req: list[list[int]] = []
        for c in candidates:
            req = c["req"]
            row_len = 1 + (
                k if c["verify_slots"] is not None else 0
            )
            segment = out_ids[input_offsets[c["req_idx"]] : input_offsets[c["req_idx"]] + row_len]
            target_token = int(segment[0])

            if c["verify_slots"] is None:
                output_tokens_by_req.append([target_token])
                continue

            m = 0
            for j in range(k):
                if c["drafts"][j] != int(segment[j]):
                    break
                m += 1
            correction = int(segment[m])

            self.total_drafted += c["real_draft_len"]
            self.total_accepted += min(m, c["real_draft_len"])

            keep_tokens = c["base_len"] + m
            req.block_table = cache_ops.rollback_to_length(
                req.block_table, keep_tokens
            )
            req.num_blocks = len(req.block_table)
            req.slot_mapping = []

            output_tokens = c["drafts"][:m] + [correction]
            if c["remaining"] is not None:
                output_tokens = output_tokens[: c["remaining"]]
            output_tokens_by_req.append(output_tokens)

        self._record_step_stats(output_tokens_by_req)

        return output_tokens_by_req

    def _draft_prompt_lookup_tokens(self, req, num_tokens: int) -> list[int]:
        """Prompt-lookup draft：n-gram 后缀匹配。

        在请求自身序列（prompt + 已生成）中查找当前后缀的最近一次出现，
        取该出现之后的最多 num_tokens 个 token 作为 draft；命中不了返回 []。
        正确性不依赖命中质量——所有 draft 都会经过 target 模型的精确验证。
        """
        context = list(req.prompt_token_ids) + list(req.generated_token_ids)
        n = len(context)
        max_ngram = min(self.prompt_lookup_max_ngram, n - 1)
        for size in range(max_ngram, self.prompt_lookup_min_ngram - 1, -1):
            pattern = context[n - size :]
            # 从后往前找最近一次出现（排除末尾的 pattern 自身）
            for i in range(n - size - 1, -1, -1):
                if context[i : i + size] == pattern:
                    start = i + size
                    return context[start : start + num_tokens]
        return []

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
