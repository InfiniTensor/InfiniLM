import asyncio


class InferTask:
    def __init__(self, id, tokenizer, request):
        self.id_ = id
        self.finished_reason = None
        messages = request.get("messages", [])
        if len(messages) == 0:
            self.finished_reason = "invalid request"
            self.tokens = []
        else:
            input_content = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            self.tokens = tokenizer.encode(input_content)
        self.request = request
        self.output_queue = asyncio.Queue()
        self._kv_cache_pool_item = None
        self.pos = 0
        
        
    def bind_kvcache(self, kv_cache_pool_item, pos):
        self._kv_cache_pool_item = kv_cache_pool_item
        self.pos = pos
        self.tokens = self.tokens[pos:]
    
    def kvcache(self):
        return self._kv_cache_pool_item.kvcache

