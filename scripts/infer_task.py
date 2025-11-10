class InferTask:
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens):
        self.id = id
        self.finish_reason = None
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.end_tokens = end_tokens
        self._kv_cache = None
        self.pos = 0

    def bind_kvcache(self, kv_cache, pos=0):
        self._kv_cache = kv_cache
        self.pos = pos
        self.tokens = self.tokens[pos:]

    def release_kvcache(self):
        cache = self._kv_cache
        self._kv_cache = None
        return cache

    def kvcache(self):
        return self._kv_cache

    def next(self, out_token):
        self._kv_cache.update_tokens(self.tokens, self.pos)

        self.pos += len(self.tokens)
        if out_token == None or out_token in self.end_tokens:
            self.finish_reason = "stop"
        elif self.pos >= self.max_tokens:
            self.finish_reason = "length"
        else:
            self.tokens = [out_token]

    def __str__(self):
        """返回用户友好的字符串表示"""
        return (
            f"InferTask(id={self.id}, "
            f"tokens_len={len(self.tokens)}, "
            f"pos={self.pos}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}, "
            f"topk={self.topk}, "
            f"topp={self.topp}, "
            f"finish_reason={self.finish_reason}, "
            f"has_kv_cache={self._kv_cache is not None})"
        )

    def __repr__(self):
        """返回开发者友好的详细表示"""
        # 显示前10个token，避免输出过长
        tokens_preview = self.tokens[:10] if len(self.tokens) > 10 else self.tokens
        if len(self.tokens) > 10:
            tokens_preview_str = str(tokens_preview) + f" ... (total {len(self.tokens)} tokens)"
        else:
            tokens_preview_str = str(tokens_preview)

        return (
            f"InferTask(\n"
            f"  id={self.id},\n"
            f"  tokens={tokens_preview_str},\n"
            f"  pos={self.pos},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  temperature={self.temperature},\n"
            f"  topk={self.topk},\n"
            f"  topp={self.topp},\n"
            f"  end_tokens={self.end_tokens},\n"
            f"  finish_reason={self.finish_reason},\n"
            f"  has_kv_cache={self._kv_cache is not None}\n"
            f")"
        )

    def debug_info(self):
        """返回详细的调试信息"""
        return {
            "id": self.id,
            "tokens": self.tokens,
            "tokens_len": len(self.tokens),
            "pos": self.pos,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "topk": self.topk,
            "topp": self.topp,
            "end_tokens": self.end_tokens,
            "finish_reason": self.finish_reason,
            "has_kv_cache": self._kv_cache is not None,
            "remaining_tokens": self.max_tokens - self.pos
        }


class KVCache:
    def __init__(self, model):
        self._kvcache = model.create_kv_cache()
        self.tokens = [0 for _ in range(model.max_context_len())]

    def data(self):
        return self._kvcache

    def drop(self, model):
        model.drop_kv_cache(self._kvcache)

    def update_tokens(self, tokens, pos):
        end = pos + len(tokens)
        max_len = len(self.tokens)

        # If overflow, truncate tokens to fit
        if end > max_len:
            tokens = tokens[: max_len - pos]
            end = max_len

        self.tokens[pos:end] = tokens

    def __str__(self):
        """返回用户友好的字符串表示"""
        # 计算非零token数量（已使用的token）
        used_tokens = sum(1 for t in self.tokens if t != 0)
        return f"KVCache(used_tokens={used_tokens}, max_capacity={len(self.tokens)})"

    def __repr__(self):
        """返回开发者友好的详细表示"""
        # 显示前20个token，避免输出过长
        tokens_preview = self.tokens[:20] if len(self.tokens) > 20 else self.tokens
        if len(self.tokens) > 20:
            tokens_preview_str = str(tokens_preview) + f" ... (total {len(self.tokens)} slots)"
        else:
            tokens_preview_str = str(tokens_preview)

        used_tokens = sum(1 for t in self.tokens if t != 0)

        return (
            f"KVCache(\n"
            f"  tokens={tokens_preview_str},\n"
            f"  used_tokens={used_tokens},\n"
            f"  max_capacity={len(self.tokens)},\n"
            f"  usage_ratio={used_tokens/len(self.tokens):.2%}\n"
            f")"
        )
    def debug_info(self):
        """返回详细的调试信息"""
        used_tokens = sum(1 for t in self.tokens if t != 0)
        return {
            # "tokens": self.tokens,
            "total_slots": len(self.tokens),
            "used_tokens": used_tokens,
            "empty_slots": len(self.tokens) - used_tokens,
            "usage_ratio": used_tokens / len(self.tokens),
            "is_full": used_tokens >= len(self.tokens)
        }
