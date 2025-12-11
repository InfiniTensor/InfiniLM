import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import math
from typing import List
import uuid
import threading
import queue
import asyncio
from typing import Dict
import time
import collections

from icinfer.config import Config
from icinfer.sampling_params import SamplingParams
from icinfer.engine.sequence import Sequence
from icinfer.engine.scheduler import Scheduler
from icinfer.engine.model_runner import ModelRunner
from icinfer.engine.infer_task import KVCache, InferTask

import logging

logger = logging.getLogger(__name__)


class InfiniEngineAsync:

    def __init__(self, model, device, ndev, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []
        self.model_runner = ModelRunner(config, device, ndev, 0, self.events)
        self.eos_token_id = self.model_runner.eos_token_id
        self.max_context_len = self.model_runner.max_context_len()
        self.request_queue = queue.Queue()
        self.result_queues: Dict[str, asyncio.Queue] = {}
        self.main_loop = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True, trust_remote_code=kwargs["trust_remote_code"]
        )
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    async def add_request(
        self, prompt: str | list[int], sampling_params: SamplingParams, request_id: str
    ):
        if self.main_loop is None:
            self.main_loop = asyncio.get_running_loop()

        result_queue = asyncio.Queue()
        self.result_queues[request_id] = result_queue
        self.request_queue.put((prompt, sampling_params, request_id))

        return result_queue

    def add_request_action(self, prompt: str | list[int], sp, req_id):
        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = prompt

        seq = Sequence(
            prompt_tokens, sp, block_size=self.scheduler.block_size, req_id=req_id
        )
        infer_task = InferTask(
            seq.req_id,
            prompt_tokens,
            self.max_context_len,
            sp.temperature,
            sp.topk,
            sp.topp,
            self.eos_token_id,
        )
        if self.model_runner.enable_paged_attn:
            pass
        else:
            infer_task.bind_kvcache(KVCache(self.model_runner))
        seq.bind_infer_task(infer_task)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        drop_kvcache_list = self.scheduler.postprocess(seqs, token_ids)
        if self.model_runner.enable_paged_attn:
            pass
        else:
            for kv_cache in drop_kvcache_list:
                kv_cache.drop(self.model_runner)
        outputs = [
            (seq.req_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def engine_loop(self):
        while True:
            # 1. 从队列中获取新请求并添加到调度器
            while not self.request_queue.empty():
                prompt, sp, req_id = self.request_queue.get()

                self.add_request_action(prompt, sp, req_id)

                if self.request_queue.empty():
                    time.sleep(0.1)
                continue

            # 2. 执行一步推理
            if not self.scheduler.is_finished():
                seqs, is_prefill = self.scheduler.schedule()
                print(f"seqs_len: {len(seqs)}")

                # token_ids 是一个列表，按进入顺序排列的
                token_ids = self.model_runner.call("run", seqs, is_prefill)

                for seq_order_i in range(len(seqs)):
                    seq = seqs[seq_order_i]
                    new_token = token_ids[seq_order_i]
                    result_queue = self.result_queues.get(seq.req_id)
                    if result_queue:
                        self.main_loop.call_soon_threadsafe(
                            result_queue.put_nowait, new_token
                        )

                drop_kvcache_list = self.scheduler.postprocess(seqs, token_ids)
                if self.model_runner.enable_paged_attn:
                    pass
                else:
                    for kv_cache in drop_kvcache_list:
                        kv_cache.drop(self.model_runner)

                # 4. 处理完成的序列
                for seq in seqs:
                    if seq.is_finished:
                        result_queue = self.result_queues.get(seq.req_id)
                        if result_queue:
                            self.main_loop.call_soon_threadsafe(
                                result_queue.put_nowait, None
                            )
                        self.result_queues.pop(seq.req_id, None)
            else:
                time.sleep(0.01)
