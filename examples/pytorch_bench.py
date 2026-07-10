import argparse
import os
import sys
import time
from types import MethodType

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--devices", default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.devices
    os.environ.setdefault("HIP_VISIBLE_DEVICES", _pre_args.devices)

import torch
import transformers.utils.import_utils as import_utils
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, new_seq_length=None, layer_idx=0: self.get_seq_length(layer_idx)
except Exception:
    pass
if not hasattr(import_utils, "is_torch_fx_available"):
    def is_torch_fx_available():
        return hasattr(torch, "fx")
    import_utils.is_torch_fx_available = is_torch_fx_available
CASES = {
    "bs4_1024_1024": (4, 1024, 1024),
    "bs4_4096_4096": (4, 4096, 4096),
    "bs16_128_128": (16, 128, 128),
    "bs16_1024_1024": (16, 1024, 1024),
    "bs16_4096_4096": (16, 4096, 4096),
}
def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
def make_inputs(tokenizer, batch_size, input_len, device):
    prompt = "Hello"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    ids = tokenizer.encode(prompt)
    ids = (ids * ((input_len + len(ids) - 1) // len(ids)))[:input_len]
    input_ids = torch.tensor([ids] * batch_size, dtype=torch.long, device=device)
    return input_ids, torch.ones_like(input_ids)
def normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)
def input_device_for_model(model, fallback):
    device_map = getattr(model, "hf_device_map", None) or {}
    for key in ("model.embed_tokens", "model.norm", ""):
        if key in device_map:
            return normalize_device(device_map[key])
    try:
        return next(model.parameters()).device
    except StopIteration:
        return normalize_device(fallback)
def patch_last_token_logits(model):
    def forward_last_token_logits(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if labels is not None:
            raise NotImplementedError("last-token logits patch is only for generation, not loss computation")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0][:, -1:, :]
        logits = self.lm_head(hidden_states).float()
        if input_ids is not None and logits.device != input_ids.device:
            logits = logits.to(input_ids.device, non_blocking=True)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = MethodType(forward_last_token_logits, model)
    return model
def parse_max_memory(spec):
    max_memory = {}
    for item in spec.split(","):
        device, memory = item.split(":", 1)
        device = device.strip()
        max_memory[int(device) if device.isdigit() else device] = memory.strip()
    return max_memory
def build_dsv2_lmhead0_device_map(model_path, split_layer=13):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = getattr(config, "num_hidden_layers")
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
    }
    for layer_idx in range(num_layers):
        device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx < split_layer else 1
    return device_map
def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
def manual_greedy_generate(model, input_ids, attention_mask, output_len, tokenizer, progress_step):
    generated = input_ids
    past_key_values = None
    next_input_ids = input_ids
    cur_attention_mask = attention_mask
    start = time.time()
    pad_token_id = tokenizer.eos_token_id

    for step in range(output_len):
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=cur_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        next_tokens = next_tokens.to(generated.device)
        generated = torch.cat([generated, next_tokens[:, None]], dim=-1)
        next_input_ids = next_tokens[:, None]
        cur_attention_mask = torch.cat(
            [
                cur_attention_mask,
                torch.ones((cur_attention_mask.shape[0], 1), dtype=cur_attention_mask.dtype, device=cur_attention_mask.device),
            ],
            dim=-1,
        )
        if progress_step and ((step + 1) % progress_step == 0 or step + 1 == output_len):
            sync_device()
            elapsed = time.time() - start
            tok_s = generated.shape[0] * (step + 1) / elapsed
            log(f"generated={step + 1}/{output_len} elapsed={elapsed:.2f}s throughput={tok_s:.2f} tok/s")
    return generated

def timed_generate(model, input_ids, attention_mask, output_len, tokenizer, progress_step=0):
    sync_device()
    t0 = time.time()
    with torch.inference_mode():
        if progress_step:
            out = manual_greedy_generate(model, input_ids, attention_mask, output_len, tokenizer, progress_step)
        else:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_len,
                min_new_tokens=output_len,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[],
            )
    sync_device()
    total_ms = (time.time() - t0) * 1000
    new_tokens = out.shape[1] - input_ids.shape[1]
    return total_ms, new_tokens
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        #default="/data-aisoft/mechdancer/models/deepseek-ai_DeepSeek-V2-Lite-Chat/",
        default="/home_aclsylqidf/shared/DeepSeek-V2-Lite-Chat/",
    )
    parser.add_argument("--case", choices=CASES.keys())
    parser.add_argument("--case-custom", nargs=3, type=int, metavar=("BATCH", "INPUT_LEN", "OUTPUT_LEN"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", default=None, help="comma-separated visible CUDA devices, e.g. 0,1")
    parser.add_argument("--device-map", default=None, help="Transformers device_map value, e.g. auto, balanced, or dsv2_lmhead0")
    parser.add_argument("--split-layer", type=int, default=13, help="first layer placed on device 1 for dsv2_lmhead0")
    parser.add_argument("--max-memory", default=None, help="per-device max memory for auto mapping, e.g. 0:60GiB,1:60GiB")
    parser.add_argument("--attn-implementation", default=None, help="attention implementation, e.g. flash_attention_2")
    parser.add_argument("--last-token-logits", action="store_true", help="compute logits only for the final token during generation")
    parser.add_argument("--progress-step", type=int, default=0, help="print decode progress every N generated tokens; uses manual greedy decode")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--cpu-to-device", action="store_true", help="load on CPU first, then model.to(device)")
    args = parser.parse_args()
    if args.case_custom:
        batch_size, input_len, output_len = args.case_custom
        args.case = f"bs{batch_size}_{input_len}_{output_len}"
    elif args.case:
        batch_size, input_len, output_len = CASES[args.case]
    else:
        parser.error("one of --case or --case-custom is required")
    log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log(f"visible device {i} name={torch.cuda.get_device_name(i)}")
    log(f"loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    log("loading model")
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if not args.cpu_to_device:
        if args.device_map:
            if args.device_map == "dsv2_lmhead0":
                load_kwargs["device_map"] = build_dsv2_lmhead0_device_map(args.model, args.split_layer)
            else:
                load_kwargs["device_map"] = args.device_map
            if args.max_memory:
                load_kwargs["max_memory"] = parse_max_memory(args.max_memory)
        else:
            load_kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
    if args.cpu_to_device:
        log(f"moving model to {args.device}")
        model = model.to(args.device).eval()
    if args.last_token_logits:
        log("patching model.forward to compute last-token logits only")
        model = patch_last_token_logits(model)
    if getattr(model, "hf_device_map", None):
        log(f"hf_device_map={model.hf_device_map}")
    log("model ready")
    log("building inputs")
    input_device = input_device_for_model(model, args.device)
    log(f"input_device={input_device}")
    input_ids, attention_mask = make_inputs(tokenizer, batch_size, input_len, input_device)
    if not args.no_warmup:
        log("warmup start")
        timed_generate(model, input_ids, attention_mask, 5, tokenizer)
        log("warmup done")
    log("benchmark start")
    total_ms, new_tokens = timed_generate(model, input_ids, attention_mask, output_len, tokenizer, args.progress_step)
    tok_s = batch_size * new_tokens / (total_ms / 1000)
    print(f"case={args.case}")
    print(f"batch_size={batch_size} input_len={input_len} output_len={output_len}")
    print(f"total_time: {total_ms:.2f} ms")
    print(f"generated_tokens_per_seq: {new_tokens}")
    print(f"decode/output throughput: {tok_s:.2f} tok/s")
if __name__ == "__main__":
    main()
