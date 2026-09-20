import ctypes
import os
import unittest

import numpy as np
import torch

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


MODEL_PATH = os.getenv("MAMBA2_MODEL_PATH")
RUN_GPU_TESTS = os.getenv("INFINILM_RUN_GPU_TESTS") == "1"


@unittest.skipUnless(
    RUN_GPU_TESTS and MODEL_PATH and torch.cuda.is_available(),
    "set INFINILM_RUN_GPU_TESTS=1 and MAMBA2_MODEL_PATH on an NVIDIA host",
)
class Mamba2RealModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferEngine(
            MODEL_PATH,
            device=infinicore.device("cuda", 0),
            distributed_config=DistConfig(1),
            cache_config=PagedKVCacheConfig(
                num_blocks=64, block_size=16, max_batch_size=4
            ),
            attention_backend="paged-attn",
            weight_load_mode="sync",
        )
        load_model_state_dict_by_file(cls.engine, MODEL_PATH, dtype=infinicore.float16)

    @classmethod
    def tearDownClass(cls):
        del cls.engine

    def _forward(self, tokens, initial_row, final_row, past_length):
        length = len(tokens)
        output = self.engine.forward_raw(
            infinicore.from_list([tokens], dtype=infinicore.int64),
            position_ids=infinicore.from_list(
                [list(range(past_length, past_length + length))],
                dtype=infinicore.int64,
            ),
            past_kv_lengths=infinicore.from_list([past_length], dtype=infinicore.int32),
            total_kv_lengths=infinicore.from_list(
                [past_length + length], dtype=infinicore.int32
            ),
            input_offsets=infinicore.from_list([0, length], dtype=infinicore.int32),
            cu_seqlens=infinicore.from_list(
                [0, past_length + length], dtype=infinicore.int32
            ),
            mamba_init_state_indices=infinicore.from_list(
                [initial_row], dtype=infinicore.int32
            ),
            mamba_final_state_indices=infinicore.from_list(
                [final_row], dtype=infinicore.int32
            ),
            sample_all_positions=True,
        )
        logits = output["logits"].to(infinicore.device("cpu", 0))
        values = np.empty(logits.shape, dtype=np.float16)
        ctypes.memmove(values.ctypes.data, logits.data_ptr(), values.nbytes)
        return values

    def test_load_prefill_decode_and_state_isolation(self):
        prefill = self._forward([1, 2, 3, 4, 5, 6, 7, 8], 0, 1, 0)
        decode = self._forward([9], 1, 1, 8)
        parallel = self._forward([10, 11], 0, 2, 0)

        self.assertEqual(prefill.shape, (1, 8, 50288))
        self.assertEqual(decode.shape, (1, 1, 50288))
        self.assertEqual(parallel.shape, (1, 2, 50288))
        self.assertTrue(np.isfinite(prefill).all())
        self.assertTrue(np.isfinite(decode).all())
        self.assertTrue(np.isfinite(parallel).all())


if __name__ == "__main__":
    unittest.main()
