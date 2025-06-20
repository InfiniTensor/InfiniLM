import asyncio
from typing import List
from libinfinicore_infer import create_kv_cache, POINTER, KVCache
from jiuge import JiugeForCauslLM, RandSampleArgs, RequestMeta


class ModelRunner:
    def __init__(self, model: JiugeForCauslLM):
        self.model = model
        self.queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.max_reqs = 4
        
    async def request(self, request_dict, request_id):
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
        # TODO: 
        kv_cache = create_kv_cache(self.model.model_instance)
        # Build requestMeta
        req = RequestMeta(
            id=request_id,
            tokens=tokens,
            args=args,
            request=request_dict,
            kv_cache=kv_cache,
            pos=0
        )
        
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
            except asyncio.QueueEmpty:
                pass
            
            # Do inference for reqs
            try:
                self.model.infer(reqs)
            except Exception as e:
                print(f"model.infer error: {e}")
                continue
        
            # Finish inference
            for r in reqs:
                output_str = self.model.tokenizer.decode(r.outputs)
                print(f"[Result for {r.id}]: {output_str.strip()}")

       
async def test_runner():
    import sys
    from libinfinicore_infer import DeviceType
    model_path = sys.argv[2]
    device = DeviceType.DEVICE_TYPE_ASCEND  # 改为你的设备
    model = JiugeForCauslLM(model_path, device, ndev=1)
    runner = ModelRunner(model)

    # 启动 batch loop
    asyncio.create_task(runner.service())

    # 模拟提交多个请求
    prompts = ["你好", "介绍一下人工智能", "太阳有多大"]
    for i, p in enumerate(prompts):
        req_dict = {
            "model": "jiuge",
            "messages": [
            {"role": "user", "content": p}
            ],
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.8,
            "max_tokens": 512,
            "stream": True
        }
        await runner.request(req_dict, request_id=i)

    await asyncio.sleep(2)  # 等待结果
    model.destroy_model_instance()
    

if __name__ == "__main__":
    asyncio.run(test_runner())
