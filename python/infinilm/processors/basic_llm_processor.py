from .processor import InfinilmProcessor
from transformers import AutoTokenizer
from ..llm.static_scheduler import StaticSchedulerOutput
from ..llm.scheduler import SchedulerOutput


class BasicLLMProcessor(InfinilmProcessor):
    def __init__(self, model_dir_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )

    def __call__(self, prompt: str, return_tensors: str = None, **kwargs) -> dict:
        if return_tensors is None:
            return self.tokenizer(prompt)
        elif return_tensors == "infini":
            import infinilm.core as infinicore

            result = {}
            for key, tensor in self.tokenizer(prompt, return_tensors="pt").items():
                result[key] = tensor.from_torch(tensor)
            return result

        # "pt" or "np" or "tf".
        return self.tokenizer(prompt, return_tensors="pt")

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ):
        normalized_conversation = []
        for message in conversation:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1, "Only one content item supported in list"
                content_item = message["content"][0]
                assert "type" in content_item and "text" in content_item, "Content dict must have 'type' and 'text' keys"
                normalized_conversation.append(
                    {"role": message["role"], "content": content_item["text"]}
                )
            else:
                normalized_conversation.append(message)
        return self.tokenizer.apply_chat_template(
            conversation=normalized_conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )

    def build_model_inputs(
        self,
        scheduler_output: SchedulerOutput | StaticSchedulerOutput,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 1,
    ) -> dict:
        """Process a batch of data and return a dictionary of model inputs."""
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
        self, scheduler_output: StaticSchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        """Construct model inputs for prefill or decode phase.

        Static cache model inputs:

        Prefill phase (with prefix cache reuse):
            - input_ids: Tokens after the cached prefix [1, prompt_length - prefix_hit_len]
            - position_ids: [prefix_hit_len, ..., prompt_length-1]
            - past_kv_lengths: [prefix_hit_len]  (reuse cached prefix)
            - total_kv_lengths: [prompt_length]

        Decode phase:
            - input_ids: Only the last generated token [1, 1]
            - position_ids: [current_position] (position in full sequence)
            - past_kv_lengths: [num_cached_tokens]
            - total_kv_lengths: [total_tokens]
        """
        import infinilm.core as infinicore

        """Build model input from static scheduler output."""
        req = scheduler_output.scheduled_requests[0]

        if scheduler_output.is_prefill:
            # Prefill: only send tokens not already in cache
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
        self, scheduler_output: SchedulerOutput, temperature, top_p, top_k
    ) -> dict:
        """Construct model inputs for prefill or decode phase.

        Prefill phase:
            - input_ids: Flattened token list (excluding cached tokens)
            - position_ids: Position IDs for new tokens in complete sequence
            - past_kv_lengths: Number of cached tokens per request
            - total_kv_lengths: Total tokens (cached + new) per request
            - input_offsets: Start position of each request in flattened array
            - block_tables: Padded block_table for each request
            - slot_mapping: Token to slot mappings

        Decode phase:
            - input_ids: Only last generated token per request
            - position_ids: Position of last token in complete sequence
            - past_kv_lengths: Number of cached tokens per request
            - total_kv_lengths: Total sequence length per request
            - input_offsets: Offsets for each request
            - block_tables: Padded block_table for each request
            - slot_mapping: Single slot per request
        """
        import infinilm.core as infinicore

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

    def get_tokenizer(self):
        return self.tokenizer
