class Qwen3KVCache:
    """Qwen3 专用的 KV 缓存包装器"""
    def __init__(self, model_instance):
        self._kvcache = model_instance.create_kv_cache()
        self.tokens = [0 for _ in range(40960)]  # max context length
        self.model_instance = model_instance

    def data(self):
        return self._kvcache

    def drop(self):
        if self._kvcache:
            # 修复：应该调用 model_instance 的 drop_kv_cache 方法
            self.model_instance.drop_kv_cache(self._kvcache)
            self._kvcache = None


    def update_tokens(self, tokens, pos):
        end = pos + len(tokens)
        max_len = len(self.tokens)

        # If overflow, truncate tokens to fit
        if end > max_len:
            tokens = tokens[: max_len - pos]
            end = max_len

        self.tokens[pos:end] = tokens


class Qwen3InferTask:
    """Qwen3 专用的推理任务"""
    def __init__(self, tokens, position=0, temperature=1.0, topk=1, topp=1.0, 
                 end_tokens=None, max_tokens=40960, task_id=0):
        self.id = task_id
        self.finish_reason = None
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.end_tokens = end_tokens or [151645]  # Qwen3 eos_token_id
        self._kv_cache = None
        self.pos = position

    def release_kvcache(self):
        cache = self._kv_cache
        self._kv_cache = None
        return cache

    def kvcache(self):
        return self._kv_cache

    def next(self, out_token):
        # 修复前的问题：KV缓存更新时机和方式不正确
        if self._kv_cache:
            # 修复：应该在添加新token前更新KV缓存
            self._kv_cache.update_tokens(self.tokens, self.pos)

        # 修复：正确更新位置和tokens
        self.pos += len(self.tokens)
        
        if out_token is None or out_token in self.end_tokens:
            self.finish_reason = "stop"
        elif self.pos >= self.max_tokens:
            self.finish_reason = "length"
        else:
            # 修复：为下一轮推理准备新的token序列
            self.tokens = [out_token]
            
    def bind_kvcache(self, kv_cache, pos=0):
        self._kv_cache = kv_cache
        self.pos = pos
        # 修复：如果pos > 0，需要正确处理token序列
        if pos > 0:
            # 确保KV缓存中已有pos长度的历史tokens
            if len(self.tokens) > pos:
                self.tokens = self.tokens[pos:]
            else:
                # 如果tokens不足，这是一个错误状态
                raise ValueError(f"Token sequence length {len(self.tokens)} is less than position {pos}")

