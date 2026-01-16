class InferTask:
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens, repetition_penalty=1.0):
        self.id = id
        self.finish_reason = None
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.repetition_penalty = repetition_penalty
        self.end_tokens = end_tokens
        self._kv_cache = None
        self.pos = 0

        # vLLM-style unique token tracking for efficient repetition penalty
        # Track unique token IDs that have been generated (not the full sequence)
        # Initialize with prompt tokens so they are also penalized
        self._unique_generated_tokens = set(tokens)  # Initialize with prompt tokens!
        self._unique_tokens_array = sorted(self._unique_generated_tokens)  # Pre-sort for efficiency
        self._unique_tokens_dirty = False  # Already initialized, no need to rebuild

    def bind_kvcache(self, kv_cache, pos=0):
        self._kv_cache = kv_cache
        self.pos = pos
        # Update tokens and add any new tokens to unique set
        remaining_tokens = self.tokens[pos:]
        for token in remaining_tokens:
            if token not in self._unique_generated_tokens:
                self._unique_generated_tokens.add(token)
                self._unique_tokens_dirty = True
        self.tokens = remaining_tokens

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
            # Incrementally update unique token set (vLLM-style)
            # Only add if it's a new token (O(1) average)
            if out_token not in self._unique_generated_tokens:
                self._unique_generated_tokens.add(out_token)
                self._unique_tokens_dirty = True

    def get_unique_previous_tokens(self):
        """
        Returns a sorted list of unique token IDs that have been generated.
        This is the vLLM-style "seen tokens" list for efficient repetition penalty.

        Returns:
            tuple: (array, length) where array is sorted list of unique token IDs
        """
        if self._unique_tokens_dirty:
            self._unique_tokens_array = sorted(self._unique_generated_tokens)
            self._unique_tokens_dirty = False
        return self._unique_tokens_array, len(self._unique_tokens_array)


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
