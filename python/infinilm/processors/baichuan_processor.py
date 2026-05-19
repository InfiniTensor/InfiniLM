# python/infinilm/processors/baichuan_processor.py

import json
import os
from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("baichuan")
class BaichuanProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        self.model_dir_path = model_dir_path
        super().__init__(model_dir_path)
        self._fix_missing_chat_template()

    def _fix_missing_chat_template(self):
        """
        Baichuan models do not ship with tokenizer.chat_template in HuggingFace format.
        They use user_token_id and assistant_token_id (from generation_config.json)
        via custom build_chat_input code.

        We dynamically inject a Jinja template that reproduces the official logic.
        The key insight: the token IDs (e.g., 195, 196) must be converted to their
        actual vocabulary text via tokenizer.convert_ids_to_tokens() before being
        embedded in the template string. This is because the tokenizer's encode()
        function maps text → IDs, and the mapping is:

            generation_config.json           tokenizer vocab
            ─────────────────────          ──────────────────
            user_token_id: 195      →     "<reserved_106>"    (convert_ids_to_tokens(195))
            assistant_token_id: 196  →     "<reserved_107>"    (convert_ids_to_tokens(196))

        The string "<reserved_195>" is NOT a valid token in the vocabulary — it gets
        split into multiple garbage tokens by the tokenizer. Only the actual vocab
        text (e.g., "<reserved_106>") will correctly encode back to ID 195.

        Since different Baichuan model versions may have different token IDs and
        different vocab text, we read generation_config.json and resolve the text
        dynamically rather than hardcoding.

        You can verify the ID-to-text mapping with the following experiment:

            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("/data/rubik/models/Baichuan2-7B-Chat", trust_remote_code=True)
            print("195 →", tok.convert_ids_to_tokens(195), "→ encode:", tok.encode(tok.convert_ids_to_tokens(195)))
            print("196 →", tok.convert_ids_to_tokens(196), "→ encode:", tok.encode(tok.convert_ids_to_tokens(196)))

        Output:

            195 → <reserved_106> → encode: [195]
            196 → <reserved_107> → encode: [196]

        This confirms that "<reserved_106>" encodes to [195] and "<reserved_107>" encodes
        to [196], while the naive "<reserved_195>" would be split into garbage tokens.
        """
        if getattr(self.tokenizer, 'chat_template', None):
            return

        # Step 1: Read role token IDs from generation_config.json
        gen_config_path = os.path.join(self.model_dir_path, "generation_config.json")
        if not os.path.exists(gen_config_path):
            return

        with open(gen_config_path) as f:
            gen_config = json.load(f)

        user_token_id = gen_config.get("user_token_id")
        assistant_token_id = gen_config.get("assistant_token_id")

        if user_token_id is None or assistant_token_id is None:
            return

        # Step 2: Resolve token IDs to their actual vocabulary text
        # e.g., 195 → "<reserved_106>", 196 → "<reserved_107>"
        # These are the strings the tokenizer recognizes as single tokens.
        user_token_text = self.tokenizer.convert_ids_to_tokens(user_token_id)
        assistant_token_text = self.tokenizer.convert_ids_to_tokens(assistant_token_id)

        # Step 3: Build Jinja template using the resolved text
        baichuan_template = (
            "{%- for message in messages -%}"
            "{%- if message['role'] == 'user' -%}"
            f"{user_token_text}{{{{ message['content'] }}}}{assistant_token_text}"
            "{%- elif message['role'] == 'assistant' -%}"
            "{{ message['content'] }}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )
        self.tokenizer.chat_template = baichuan_template

