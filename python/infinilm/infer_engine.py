import time
from dataclasses import dataclass

import infinicore
import numpy as np

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm

from .modeling_utils import parse_dtype
from .exception_utils import handle_oom_and_exit


@dataclass
class GenerationConfig:
    max_new_tokens: int | None = None

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    eos_token_id: list[int] | None = None
    stop_on_eos: bool = True


class InferEngine(_infinilm.InferEngine):
    def __init__(
        self,
        model_path,
        device=None,
        distributed_config=DistConfig(1),
        cache_config=None,
        enable_graph_compiling=False,
        attention_backend="default",
        kv_cache_dtype=None,
    ):
        self.config = AutoConfig.from_pretrained(model_path)

        if device is None:
            device = infinicore.device()

        super().__init__(
            model_path,
            distributed_config._underlying,
            device._underlying.type,
            cache_config,
            enable_graph_compiling,
            attention_backend,
            (
                parse_dtype(kv_cache_dtype)._underlying
                if kv_cache_dtype is not None
                else None
            ),
        )
        self.use_cache = False

        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        *,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        cu_seqlens=None,
        block_tables=None,
        slot_mapping=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        try:
            # TODO: Remove `_underlying` and simplify the corresponding code.
            input_ids = input_ids._underlying if input_ids is not None else None
            position_ids = (
                position_ids._underlying if position_ids is not None else None
            )
            past_kv_lengths = (
                past_kv_lengths._underlying if past_kv_lengths is not None else None
            )
            total_kv_lengths = (
                total_kv_lengths._underlying if total_kv_lengths is not None else None
            )
            input_offsets = (
                input_offsets._underlying if input_offsets is not None else None
            )
            block_tables = (
                block_tables._underlying if block_tables is not None else None
            )
            cu_seqlens = cu_seqlens._underlying if cu_seqlens is not None else None
            slot_mapping = (
                slot_mapping._underlying if slot_mapping is not None else None
            )

            out = super().forward(
                super().Input(
                    input_ids,
                    position_ids=position_ids,
                    past_sequence_lengths=past_kv_lengths,
                    total_sequence_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    cu_seqlens=cu_seqlens,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
            return infinicore.Tensor(out.output_ids)
        except BaseException as e:
            handle_oom_and_exit(e)
            raise

    def forward_output(
        self,
        input_ids,
        *,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        cu_seqlens=None,
        block_tables=None,
        slot_mapping=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        try:
            input_ids = input_ids._underlying if input_ids is not None else None
            position_ids = (
                position_ids._underlying if position_ids is not None else None
            )
            past_kv_lengths = (
                past_kv_lengths._underlying if past_kv_lengths is not None else None
            )
            total_kv_lengths = (
                total_kv_lengths._underlying if total_kv_lengths is not None else None
            )
            input_offsets = (
                input_offsets._underlying if input_offsets is not None else None
            )
            block_tables = (
                block_tables._underlying if block_tables is not None else None
            )
            cu_seqlens = cu_seqlens._underlying if cu_seqlens is not None else None
            slot_mapping = (
                slot_mapping._underlying if slot_mapping is not None else None
            )

            return super().forward(
                super().Input(
                    input_ids,
                    position_ids=position_ids,
                    past_sequence_lengths=past_kv_lengths,
                    total_sequence_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    cu_seqlens=cu_seqlens,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
        except BaseException as e:
            handle_oom_and_exit(e)
            raise

    def generate(
        self,
        input_ids,
        generation_config,
        *,
        _measure_and_log_time=False,
        _return_time_measurements: bool = False,
        _return_step_breakdown: bool = False,
    ):
        def _copy_from_numpy(dst: infinicore.Tensor, src_np: np.ndarray):
            # src_np must remain alive for the duration of this call.
            tmp = infinicore.from_blob(
                int(src_np.ctypes.data),
                list(src_np.shape),
                dtype=dst.dtype,
                device=infinicore.device("cpu", 0),
            )
            dst.copy_(tmp)

        if generation_config.eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        else:
            eos_token_id = generation_config.eos_token_id

        past_seq_len = 0
        output_ids = []
        initial_batch_size, initial_seqlen = input_ids.shape[:2]
        seq_len = initial_seqlen
        batch_size = initial_batch_size

        if batch_size != 1 and generation_config.max_new_tokens is None:
            raise ValueError(
                "When `batch_size > 1`, `max_new_tokens` must be specified."
            )

        if _measure_and_log_time:
            time_measurements = []
            cpu_prep_s = []
            gpu_forward_ms = []
            gpu_sampling_ms = []
            gpu_d2h_ms = []

        block_tables = None
        max_blocks_per_batch = 0
        if self.enable_paged_attn:
            paged_block_size = self.get_cache_config().block_size()
            max_blocks_per_batch = (
                initial_seqlen + generation_config.max_new_tokens + paged_block_size - 1
            ) // paged_block_size

            block_tables_list = [
                range(i * max_blocks_per_batch, (i + 1) * max_blocks_per_batch)
                for i in range(batch_size)
            ]
            block_tables = infinicore.from_list(
                block_tables_list,
                dtype=infinicore.int32,
            )

        # Reuse per-step metadata buffers to reduce Python overhead.
        cpu = infinicore.device("cpu", 0)
        past_kv_np = np.empty((initial_batch_size,), dtype=np.int32)
        total_kv_np = np.empty((initial_batch_size,), dtype=np.int32)
        cu_seqlens_np = np.empty((initial_batch_size + 1,), dtype=np.int32)
        input_offsets_np = np.empty((initial_batch_size + 1,), dtype=np.int32)

        past_kv_lengths_buf = infinicore.empty(
            [initial_batch_size], dtype=infinicore.int32, device=cpu
        )
        total_kv_lengths_buf = infinicore.empty(
            [initial_batch_size], dtype=infinicore.int32, device=cpu
        )
        cu_seqlens_buf = infinicore.empty(
            [initial_batch_size + 1], dtype=infinicore.int32, device=cpu
        )
        input_offsets_buf = infinicore.empty(
            [initial_batch_size + 1], dtype=infinicore.int32, device=cpu
        )

        if self.enable_paged_attn:
            # Iter=0 uses the full prompt; later steps use seq_len=1.
            max_pos_elems = initial_batch_size * initial_seqlen
            position_ids_np = np.empty((max_pos_elems,), dtype=np.int64)
            slot_mapping_np = np.empty((max_pos_elems,), dtype=np.int64)
            position_ids_buf = infinicore.empty(
                [max_pos_elems], dtype=infinicore.int64, device=cpu
            )
            slot_mapping_buf = infinicore.empty(
                [max_pos_elems], dtype=infinicore.int64, device=cpu
            )
        else:
            position_ids_np = np.empty(
                (initial_batch_size, initial_seqlen), dtype=np.int64
            )
            position_ids_buf = infinicore.empty(
                [initial_batch_size, initial_seqlen],
                dtype=infinicore.int64,
                device=cpu,
            )

        for iter in range(0, generation_config.max_new_tokens):
            if _measure_and_log_time:
                start_time = time.perf_counter()

            batch_size, seq_len = input_ids.shape[:2]

            if _measure_and_log_time:
                t_prep0 = time.perf_counter()

            if self.enable_paged_attn:
                input_ids = input_ids.view([1, batch_size * seq_len])
                # position_ids: shape [batch_size * seq_len]
                cur_pos_elems = batch_size * seq_len
                if cur_pos_elems > position_ids_buf.numel():
                    raise ValueError(
                        f"position_ids buffer too small: need {cur_pos_elems}, "
                        f"have {position_ids_buf.numel()} (batch={batch_size}, seq={seq_len})"
                    )

                if seq_len == 1:
                    position_ids_np[:cur_pos_elems] = past_seq_len
                else:
                    base = np.arange(
                        past_seq_len, past_seq_len + seq_len, dtype=np.int64
                    )
                    position_ids_np[:cur_pos_elems] = np.tile(base, batch_size)

                position_ids = position_ids_buf.narrow(0, 0, cur_pos_elems)
                _copy_from_numpy(position_ids, position_ids_np[:cur_pos_elems])

                if iter == 0:
                    # slot_mapping: shape [batch_size * seq_len]
                    stride = max_blocks_per_batch * paged_block_size
                    if seq_len == 1:
                        slot_mapping_np[:cur_pos_elems] = (
                            np.arange(batch_size, dtype=np.int64) * stride
                        )
                    else:
                        base = np.arange(seq_len, dtype=np.int64)
                        slot_mapping_np[:cur_pos_elems] = (
                            np.repeat(np.arange(batch_size, dtype=np.int64), seq_len)
                            * stride
                            + np.tile(base, batch_size)
                        )
                    slot_mapping = slot_mapping_buf.narrow(0, 0, cur_pos_elems)
                    _copy_from_numpy(slot_mapping, slot_mapping_np[:cur_pos_elems])
                else:
                    # decode (seq_len==1): one slot per batch item
                    stride = max_blocks_per_batch * paged_block_size
                    slot_mapping_np[:batch_size] = past_seq_len + np.arange(
                        batch_size, dtype=np.int64
                    ) * stride
                    slot_mapping = slot_mapping_buf.narrow(0, 0, batch_size)
                    _copy_from_numpy(slot_mapping, slot_mapping_np[:batch_size])
            else:
                # position_ids: shape [batch_size, seq_len]
                if batch_size > position_ids_buf.size(0) or seq_len > position_ids_buf.size(1):
                    raise ValueError(
                        f"position_ids buffer too small: need [{batch_size},{seq_len}], "
                        f"have {position_ids_buf.shape}"
                    )

                if seq_len == 1:
                    position_ids_np[:batch_size, 0] = past_seq_len
                else:
                    base = np.arange(
                        past_seq_len, past_seq_len + seq_len, dtype=np.int64
                    )
                    position_ids_np[:batch_size, :seq_len] = base[None, :]

                position_ids = position_ids_buf.narrow(1, 0, seq_len)
                _copy_from_numpy(position_ids, position_ids_np[:batch_size, :seq_len])

                slot_mapping = None

            past_kv_np[:batch_size] = past_seq_len
            total_kv_np[:batch_size] = past_seq_len + seq_len
            step_total = past_seq_len + seq_len
            cu_seqlens_np[: batch_size + 1] = np.arange(
                batch_size + 1, dtype=np.int32
            ) * step_total
            input_offsets_np[: batch_size + 1] = np.arange(
                batch_size + 1, dtype=np.int32
            ) * seq_len

            past_kv_lengths = past_kv_lengths_buf.narrow(0, 0, batch_size)
            total_kv_lengths = total_kv_lengths_buf.narrow(0, 0, batch_size)
            cu_seqlens = cu_seqlens_buf.narrow(0, 0, batch_size + 1)
            input_offsets = input_offsets_buf.narrow(0, 0, batch_size + 1)

            _copy_from_numpy(past_kv_lengths, past_kv_np[:batch_size])
            _copy_from_numpy(total_kv_lengths, total_kv_np[:batch_size])
            _copy_from_numpy(cu_seqlens, cu_seqlens_np[: batch_size + 1])
            _copy_from_numpy(input_offsets, input_offsets_np[: batch_size + 1])

            if _measure_and_log_time:
                cpu_prep_s.append(time.perf_counter() - t_prep0)

            out = self.forward_output(
                input_ids=input_ids,
                position_ids=position_ids,
                past_kv_lengths=past_kv_lengths,
                total_kv_lengths=total_kv_lengths,
                input_offsets=input_offsets,
                cu_seqlens=cu_seqlens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            output_id = infinicore.Tensor(out.output_ids)
            if _measure_and_log_time:
                gpu_forward_ms.append(float(getattr(out, "gpu_forward_ms", 0.0)))
                gpu_sampling_ms.append(float(getattr(out, "gpu_sampling_ms", 0.0)))
                gpu_d2h_ms.append(float(getattr(out, "gpu_d2h_ms", 0.0)))

            output_ids.append(output_id)

            if (
                initial_batch_size == 1
                and generation_config.stop_on_eos
                and generation_config.max_new_tokens is not None
                and output_id.to_numpy()[0] in eos_token_id
            ):
                break

            # start_prepare_time = time.perf_counter()
            input_ids = output_id.view([batch_size, 1])

            past_seq_len = past_seq_len + seq_len

            if _measure_and_log_time:
                end_time = time.perf_counter()

                time_measurements.append((end_time - start_time))

        if _measure_and_log_time:
            import os

            if os.getenv("INFINILM_SUPPRESS_BENCH_PRINTS") != "1":
                print(
                    f"\n\n\n Generation completed in {round(sum(time_measurements) * 1000, 2)} ms"
                )
                print(
                    f" Batchsize={initial_batch_size}  Per_Batch_Input_Len={initial_seqlen}  Per_Batch_New_Tokens={len(time_measurements)}\n"
                )
                print(
                    f" Prefill TTFT: {round(time_measurements[0] * 1000, 2)} ms  Throughput: {round((initial_batch_size * initial_seqlen) / time_measurements[0], 2)} tok/s\n",
                )
                if len(time_measurements) > 1:
                    print(
                        f" Decode  Avg ITL: {round(sum(time_measurements[1:]) * 1000 / (len(time_measurements) - 1), 2)} ms   Throughput: {round((initial_batch_size * (len(time_measurements) - 1)) / sum(time_measurements[1:]), 2)} tok/s\n",
                    )

                # Optional breakdown printout (enabled via env var, requires C++ worker timing).
                if cpu_prep_s:
                    if os.getenv("INFINILM_PROFILE_STEP_BREAKDOWN") is not None:
                        n = len(time_measurements)
                        cpu_ms = [v * 1000.0 for v in cpu_prep_s]
                        print(" Per-step breakdown (ms):")
                        for i in range(n):
                            print(
                                f"  step={i:4d} cpu_prep={cpu_ms[i]:7.3f} "
                                f"gpu_fwd={gpu_forward_ms[i]:7.3f} gpu_samp={gpu_sampling_ms[i]:7.3f} gpu_d2h={gpu_d2h_ms[i]:7.3f}"
                            )

        if _return_time_measurements:
            if not _measure_and_log_time:
                raise ValueError(
                    "`_return_time_measurements=True` requires `_measure_and_log_time=True`."
                )
            if _return_step_breakdown:
                return (
                    output_ids,
                    time_measurements,
                    {
                        "cpu_prep_s": cpu_prep_s,
                        "gpu_forward_ms": gpu_forward_ms,
                        "gpu_sampling_ms": gpu_sampling_ms,
                        "gpu_d2h_ms": gpu_d2h_ms,
                    },
                )
            return output_ids, time_measurements

        return output_ids

    def reset_cache(self, cache_config):
        infinicore.sync_device()
        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)
        super().reset_cache(cache_config)

    def state_dict_keyname(self):
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
