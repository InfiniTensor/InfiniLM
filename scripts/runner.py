import asyncio
from typing import List, Callable
from libinfinicore_infer import create_kv_cache, POINTER, KVCache
from jiuge import JiugeForCauslLM, RandSampleArgs, RequestMeta
from ctypes import c_uint, c_void_p, byref, POINTER
from libinfinicore_infer import (
    JiugeMeta,
    JiugeWeights,
    KVCache,
    DataType,
    DeviceType,
    create_jiuge_model,
    destroy_jiuge_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
)


class ModelRunner:
    def __init__(self, model: JiugeForCauslLM):
        self.model = model
        self.queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.max_reqs = 4
        self.callbacks: dict[str, Callable[[List[int]], None]] = (
            {}
        )  # ✅ 每个请求独立 callback

    async def request(
        self, request_dict, request_id, callback: Callable[[List[int]], None] = None
    ):
        # Build tokens from request
        messages = request_dict.get("messages", [])
        temperature = request_dict.get("temperature", 1.0)
        topk = request_dict.get("top_k", 1)
        topp = request_dict.get("top_p", 1.0)
        max_tokens = request_dict.get("max_tokens", 512)

        # tokenize
        input_content = self.model.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        tokens = self.model.tokenizer.encode(input_content)

        # Build args
        args = RandSampleArgs(
            max_tokens=max_tokens, temperature=temperature, topk=topk, topp=topp
        )
        # Build kv_cache
        # TODO: Use kv_cache
        kv_cache = create_kv_cache(self.model.model_instance)
        # Build requestMeta
        req = RequestMeta(
            id=request_id,
            tokens=tokens,
            args=args,
            request=request_dict,
            kv_cache=kv_cache,
            pos=0,
        )
        self.register_callback(request_id=request_id, callback=callback)

        await self.queue.put(req)

    async def service(self):
        # Collect request and call model.infer
        while True:
            reqs: List[RequestMeta] = []
            req_ = await self.queue.get()
            reqs.append(req_)

            try:
                # Collect more request
                while not self.queue.empty() and len(reqs) < self.max_reqs:
                    req_ = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    reqs.append(req_)
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                pass

            await self.infer(reqs)

    async def infer(self, reqs: List[RequestMeta]):
        flat_tokens = []
        req_lens = []
        req_poses = []
        kv_caches = []
        active_reqs = []

        for r in reqs:
            flat_tokens.extend(r.tokens)
            req_lens.append(len(r.tokens))
            req_poses.append(r.pos)
            kv_caches.append(r.kv_cache)
            active_reqs.append(r)

        max_tokens = max(r.args.max_tokens for r in reqs)

        tokens = (c_uint * len(flat_tokens))(*flat_tokens)
        req_lens = (c_uint * len(active_reqs))(*req_lens)
        req_poses = (c_uint * len(active_reqs))(*req_poses)
        kv_caches = (POINTER(KVCache) * len(active_reqs))(*kv_caches)
        ans = (c_uint * len(active_reqs))()

        for step in range(max_tokens):
            infer_batch(
                self.model.model_instance,
                tokens,
                len(tokens),
                req_lens,
                len(active_reqs),
                req_poses,
                kv_caches,
                ans,
                active_reqs[0].args.temperature,
                active_reqs[0].args.topk,
                active_reqs[0].args.topp,
            )

            output_tokens = list(ans)

            new_active = []
            next_tokens = []
            next_req_poses = []
            next_kv_caches = []

            for i, r in enumerate(active_reqs):
                token = output_tokens[i]
                if token in self.model.eos_token_id:
                    r.finished = True
                r.outputs.append(token)

                # ✅ 流式推送当前 token
                if r.id in self.callbacks:
                    self.callbacks[r.id]([token])

                if not r.finished:
                    new_active.append(r)
                    next_tokens.append(token)
                    next_req_poses.append(req_poses[i] + req_lens[i])
                    next_kv_caches.append(kv_caches[i])

            if not new_active:
                break

            active_reqs = new_active
            tokens = (c_uint * len(next_tokens))(*next_tokens)
            req_lens = (c_uint * len(next_tokens))(*([1] * len(next_tokens)))
            req_poses = (c_uint * len(next_req_poses))(*next_req_poses)
            kv_caches = (POINTER(KVCache) * len(next_kv_caches))(*next_kv_caches)

        for r in reqs:
            if r.id in self.callbacks:
                self.callbacks[r.id]([])  # 空 token 表示推理结束
                del self.callbacks[r.id]

    def register_callback(self, request_id: str, callback: Callable[[List[int]], None]):
        self.callbacks[request_id] = callback


async def test_runner():
    import sys
    from libinfinicore_infer import DeviceType

    model_path = sys.argv[2] if len(sys.argv) > 2 else "./path/to/model"
    device = DeviceType.DEVICE_TYPE_ASCEND
    model = JiugeForCauslLM(model_path, device, ndev=1)
    runner = ModelRunner(model)

    # 启动服务
    asyncio.create_task(runner.service())

    prompts = ["你好", "介绍一下人工智能", "太阳有多大"]

    async def make_request(i, prompt):
        request_id = f"req-{i}"
        req_dict = {
            "model": "jiuge",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.8,
            "max_tokens": 64,
            "stream": True,
        }

        def cb(tokens: List[int]):
            if tokens:
                print(f"[{request_id}] Got tokens:", tokens)
            else:
                print(f"[{request_id}] Finished")

        await runner.request(req_dict, request_id, callback=cb)

    # 发起所有请求
    await asyncio.gather(*(make_request(i, p) for i, p in enumerate(prompts)))

    await asyncio.sleep(3)
    model.destroy_model_instance()


if __name__ == "__main__":
    asyncio.run(test_runner())
