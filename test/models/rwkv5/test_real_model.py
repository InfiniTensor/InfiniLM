import os
import unittest

import torch

from infinilm.llm.llm import LLM
from infinilm.llm.sampling_params import SamplingParams


MODEL_PATH = os.getenv("RWKV5_MODEL_PATH")
RUN_GPU_TESTS = os.getenv("INFINILM_RUN_GPU_TESTS") == "1"


@unittest.skipUnless(
    RUN_GPU_TESTS and MODEL_PATH and torch.cuda.is_available(),
    "set INFINILM_RUN_GPU_TESTS=1 and RWKV5_MODEL_PATH on an NVIDIA host",
)
class RWKV5RealModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LLM(
            model_path=MODEL_PATH,
            device="cuda",
            dtype="bfloat16",
            tensor_parallel_size=1,
            cache_type="paged",
            max_batch_size=4,
            max_tokens=24,
            num_blocks=64,
            block_size=16,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            enable_graph=False,
            attn_backend="paged-attn",
            weight_load_mode="sync",
            enable_prefix_caching=False,
        )
        cls.sampling = SamplingParams(
            max_tokens=24,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    @staticmethod
    def _conversations():
        return [
            [{"role": "user", "content": "What is the capital of France?"}],
            [{"role": "user", "content": "Write the first three prime numbers."}],
        ]

    def test_known_answer_and_batched_state_isolation(self):
        conversations = self._conversations()
        batched = self.model.chat(
            messages=conversations,
            sampling_params=self.sampling,
            use_tqdm=False,
        )
        sequential = [
            self.model.chat(
                messages=conversation,
                sampling_params=self.sampling,
                use_tqdm=False,
            )[0]
            for conversation in conversations
        ]

        self.assertIn("Paris", batched[0].outputs[0].text)
        for batched_output, sequential_output in zip(batched, sequential):
            self.assertEqual(
                batched_output.outputs[0].token_ids,
                sequential_output.outputs[0].token_ids,
            )


if __name__ == "__main__":
    unittest.main()
