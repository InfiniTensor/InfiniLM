from .basic_llm_processor import BasicLLMProcessor
from transformers import AutoTokenizer
import json
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class BaichuanProcessor(BasicLLMProcessor):
    """Processor for Baichuan models with special chat template handling."""

    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self.user_token_id = 195
        self.assistant_token_id = 196
        # Baichuan2 uses different token IDs
        self.eos_token_id = None
        self._load_role_token_ids(model_dir_path)
        self._setup_tokenizer()

    def _load_role_token_ids(self, model_dir_path: str):
        """Load user and assistant token IDs from generation config."""
        generation_config_path = os.path.join(model_dir_path, "generation_config.json")
        if os.path.exists(generation_config_path):
            try:
                with open(generation_config_path, "r") as f:
                    generation_config = json.load(f)
                self.user_token_id = int(
                    generation_config.get("user_token_id", self.user_token_id)
                )
                self.assistant_token_id = int(
                    generation_config.get("assistant_token_id", self.assistant_token_id)
                )
                self.eos_token_id = generation_config.get("eos_token_id")
                logger.info(
                    f"Loaded Baichuan config: user={self.user_token_id}, assistant={self.assistant_token_id}, eos={self.eos_token_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to load generation config: {e}, using defaults")

    def _setup_tokenizer(self):
        """Setup tokenizer with proper special tokens."""
        # Set eos_token if not already set
        if self.tokenizer.eos_token is None:
            if self.eos_token_id is not None:
                # Try to find token for eos_token_id
                for token, token_id in self.tokenizer.get_vocab().items():
                    if token_id == self.eos_token_id:
                        self.tokenizer.eos_token = token
                        break
            if self.tokenizer.eos_token is None:
                # Default for Baichuan
                self.tokenizer.eos_token = "</s>"

        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **kwargs,
    ):
        """Apply Baichuan's special chat template.

        Baichuan uses special token IDs for user and assistant roles.
        The correct format should not add extra special tokens.
        """
        # Extract text from multimodal content if needed
        messages = []
        for message in conversation:
            role = message.get("role", "")
            content = message.get("content", "")

            # Handle multimodal content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)

            messages.append({"role": role, "content": content})

        # Use tokenizer's chat template if available
        if (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            result = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )
            return result

        # Build conversation string with proper formatting
        # This format should mimic what the tokenizer would produce
        result_parts = []
        for i, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":
                result_parts.append(f"{content}")
            elif role == "assistant":
                result_parts.append(f"{content}")
            else:
                result_parts.append(content)

            # Add separator between turns
            if i < len(messages) - 1:
                result_parts.append("\n")

        result = "".join(result_parts)

        # For generation prompts, we want the model to start generating assistant response
        # The model will naturally generate after seeing the assistant role marker
        if add_generation_prompt:
            result += "\n"

        return result

    def get_eos_token_ids(self) -> List[int]:
        """Get EOS token IDs for Baichuan model."""
        eos_ids = []

        # Add standard EOS token
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)

        # Add configured eos_token_id
        if self.eos_token_id is not None and self.eos_token_id not in eos_ids:
            eos_ids.append(self.eos_token_id)

        # Baichuan2 might use 2 as EOS
        if 2 not in eos_ids:
            eos_ids.append(2)

        # Also check for common EOS token strings
        for token, token_id in self.tokenizer.get_vocab().items():
            if token in ["</s>", "<eos>", "<|endoftext|>"] and token_id not in eos_ids:
                eos_ids.append(token_id)

        return eos_ids

    def build_model_inputs(
        self, scheduler_output, temperature=1.0, top_p=0.8, top_k=1, **kwargs
    ):
        """Build batched model inputs from scheduler output."""
        from ..llm.static_scheduler import StaticSchedulerOutput
        from ..llm.scheduler import SchedulerOutput

        if isinstance(scheduler_output, StaticSchedulerOutput):
            return self._build_model_input_from_static_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
        elif isinstance(scheduler_output, SchedulerOutput):
            return self._build_model_input_from_batch_scheduler_output(
                scheduler_output, temperature, top_p, top_k
            )
        else:
            raise ValueError(
                "scheduler_output must be an instance of SchedulerOutput or StaticSchedulerOutput"
            )

    def _build_model_input_from_static_scheduler_output(
        self, scheduler_output, temperature, top_p, top_k
    ) -> dict:
        """Build model inputs for static cache (single request)."""
        import infinicore

        req = scheduler_output.scheduled_requests[0]

        if scheduler_output.is_prefill:
            # Prefill: send tokens not already in cache
            tokens = req.get_input_tokens()
            prefix_hit_len = scheduler_output.prefix_hit_len
            input_tokens = tokens[prefix_hit_len:]
            input_ids = [input_tokens]
            position_ids = [list(range(prefix_hit_len, len(tokens)))]
            past_kv_len = prefix_hit_len
            total_kv_len = len(tokens)
            input_offsets = [0, len(input_tokens)]
        else:
            # Decode: send only the last generated token
            last_token = req.generated_token_ids[-1]
            current_position = req.get_total_length() - 1
            input_ids = [[last_token]]
            position_ids = [[current_position]]
            past_kv_len = current_position
            total_kv_len = req.get_total_length()
            input_offsets = [0, 1]

        return {
            "input_ids": infinicore.from_list(input_ids, dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(
                [past_kv_len], dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list(
                [total_kv_len], dtype=infinicore.int32
            ),
            "input_offsets": infinicore.from_list(
                input_offsets, dtype=infinicore.int32
            ),
            "cu_seqlens": infinicore.from_list(
                [0, total_kv_len], dtype=infinicore.int32
            ),
            "block_tables": None,
            "slot_mapping": None,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

    def _build_model_input_from_batch_scheduler_output(
        self, scheduler_output, temperature, top_p, top_k
    ) -> dict:
        """Build model inputs for paged cache (batch requests)."""
        import infinicore

        if not scheduler_output.scheduled_requests:
            raise RuntimeError(
                "build_model_inputs called with empty scheduled_requests"
            )

        tokens = []
        seq_lens = []
        seq_offsets = [0]
        block_tables = []
        slot_mapping = []
        cached_lens = []
        position_ids = []
        cu_seqlens = [0]

        max_block_table_len = max(
            len(req.block_table) for req in scheduler_output.scheduled_requests
        )
        current_offset = 0

        for req in scheduler_output.scheduled_requests:
            num_cached = req.num_cached_tokens
            if scheduler_output.is_prefill:
                # Prefill phase
                req_tokens = req.get_input_tokens()
                tokens_to_compute = req_tokens[num_cached:]
                tokens.extend(tokens_to_compute)

                compute_len = len(tokens_to_compute)
                seq_len = len(req_tokens)
                seq_lens.append(seq_len)

                current_offset += compute_len
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(num_cached)
                position_ids.extend(range(num_cached, num_cached + compute_len))
            else:
                # Decode phase
                seq_len = req.get_total_length()
                last_token = req.generated_token_ids[-1]
                tokens.append(last_token)
                seq_lens.append(seq_len)

                current_offset += 1
                seq_offsets.append(current_offset)

                slot_mapping.extend(req.slot_mapping)
                cached_lens.append(num_cached)
                position_ids.append(seq_len - 1)

            # Pad block_table to same length
            padded_block_table = req.block_table + [-1] * (
                max_block_table_len - len(req.block_table)
            )
            block_tables.append(padded_block_table)
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        return {
            "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
            "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list(
                cached_lens, dtype=infinicore.int32
            ),
            "total_kv_lengths": infinicore.from_list(seq_lens, dtype=infinicore.int32),
            "input_offsets": infinicore.from_list(seq_offsets, dtype=infinicore.int32),
            "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
            "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
