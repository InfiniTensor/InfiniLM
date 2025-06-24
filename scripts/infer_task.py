import janus


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
        self.output_queue = janus.Queue()
        self._kv_cache_pool_item = None
        self.pos = 0
        print(f"[INFO] Create InferTask {self.id}")

    def bind_kvcache(self, kv_cache_pool_item, pos):
        self._kv_cache_pool_item = kv_cache_pool_item
        self.pos = pos
        self.tokens = self.tokens[pos:]

    def kvcache(self):
        return self._kv_cache_pool_item.kvcache

    def output(self, out_token):
        self._kv_cache_pool_item.update_tokens(self.tokens, self.pos)

        self.pos += len(self.tokens)
        if out_token == None or out_token in self.end_tokens:
            self.finish_reason = "stop"
        elif self.pos >= self.max_tokens:
            self.finish_reason = "length"
        else:
            self.tokens = [out_token]

        self.output_queue.sync_q.put(out_token)
