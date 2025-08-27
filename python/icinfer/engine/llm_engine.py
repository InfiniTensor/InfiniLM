import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import uuid

from icinfer.config import Config
from icinfer.sampling_params import SamplingParams
from icinfer.engine.sequence import Sequence
from icinfer.engine.scheduler import Scheduler
from icinfer.engine.model_runner import ModelRunner
from icinfer.engine.infer_task import KVCache, InferTask
import logging
logger = logging.getLogger(__name__)


class InfiniEngine:

    def __init__(self, model, device, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        self.ps = []
        self.events = []
        # ctx = mp.get_context("spawn")
        # for i in range(1, config.tensor_parallel_size):
        #     event = ctx.Event()
        #     process = ctx.Process(target=ModelRunner, args=(config, i, event))
        #     process.start()
        #     self.ps.append(process)
        #     self.events.append(event)
        self.model_runner = ModelRunner(config, device, 0, self.events)
        self.eos_token_id = self.model_runner.eos_token_id
        self.max_context_len = self.model_runner.max_context_len()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=kwargs["trust_remote_code"])
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, block_size=self.scheduler.block_size)
        infer_task = InferTask(seq.seq_id, prompt, self.max_context_len, sampling_params.temperature, sampling_params.topk, sampling_params.topp, self.eos_token_id)
        if self.model_runner.enable_paged_attn:
            pass
        else:
            infer_task.bind_kvcache(KVCache(self.model_runner))
        seq.bind_infer_task(infer_task)
        self.scheduler.add(seq)
        return prompt

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        drop_kvcache_list = self.scheduler.postprocess(seqs, token_ids)
        if self.model_runner.enable_paged_attn:
            pass
        else:
            for kv_cache in drop_kvcache_list:
                kv_cache.drop(self.model_runner)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        prompts_list = []
        for prompt, sp in zip(prompts, sampling_params):
            prompts_list.append(self.add_request(prompt, sp))
        outputs = {}    
        prefill_throughput = decode_throughput = 0.
        logger.info("start generating")
        # perfile
        avg_prefill_throughput = 0
        prefill_time = 0
        avg_decode_throughput = 0
        decode_time = 0
        ttft = 0
        ttft_count = 0
        tbt = 0
        tbt_count = 0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    check_time = perf_counter()
                    prefill_throughput = num_tokens / (check_time - t)
                    ttft += (check_time - t)
                    ttft_count += 1
                    avg_prefill_throughput = (avg_prefill_throughput * prefill_time + num_tokens)/(prefill_time+(check_time - t))
                    prefill_time += (check_time - t)
                else:
                    check_time = perf_counter()
                    decode_throughput = -num_tokens / (check_time - t)
                    tbt += (check_time - t)
                    tbt_count += 1
                    avg_decode_throughput = (avg_decode_throughput * decode_time - num_tokens)/(decode_time+(check_time - t))
                    decode_time += (check_time - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        avg_ttft = ttft / ttft_count
        avg_tbt = tbt / tbt_count
        if not self.model_runner.enable_paged_attn:
            max_model_len = self.model_runner.config.max_model_len
            num_seqs = len(outputs)
            used_tokens = [len(prompts_list[i])+len(outputs[i]) for i in range(num_seqs)]
            used_tokens_count = sum(used_tokens)
            cache_efficiency = used_tokens_count / (num_seqs * max_model_len)
        else:
            max_model_len = self.model_runner.config.max_model_len
            num_seqs = len(outputs)
            used_tokens = [len(prompts_list[i])+len(outputs[i]) for i in range(num_seqs)]
            block_size = self.model_runner.config.kvcache_block_size
            cache_memory = [(i_tokens + block_size - 1) // block_size * block_size for i_tokens in used_tokens]
            cache_efficiency = sum(used_tokens) / sum(cache_memory)
        
        if use_tqdm:
            pbar.close()
        return outputs, avg_prefill_throughput, avg_decode_throughput, avg_ttft, avg_tbt, cache_efficiency