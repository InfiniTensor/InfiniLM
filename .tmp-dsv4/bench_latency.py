#!/usr/bin/env python3
import argparse, concurrent.futures, hashlib, json, statistics, threading, time, urllib.request

from transformers import AutoTokenize

MODEL = os.environ.get("DSV4_MODEL_PATH", "/home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp")

CORPUS = (
    "大模型推理系统在昇腾九百一十C集群上运行时，需要同时关心算子内核效率、通信重叠、内存水位与调度策略。"
    "张量并行会把注意力头和专家权重切分到不同设备，MLA的潜空间缓存决定了长上下文的容量上限，"
    "MoE路由则要求令牌重排与分组矩阵乘法紧密配合。图模式可以显著降低每步下发开销，"
    "但捕获期固化的任何宿主侧状态都必须在重放前刷新，否则会在长序列上逐渐放大误差。"
    "分页KV缓存以块为单位管理生命周期，前缀命中、块分配与释放策略都会影响尾延迟。"
    "在真实服务里，客户端并发、输入长度和输出长度共同决定了预填充与解码阶段的占比，"
    "因此基准测试必须固定温度、输入token数、输出token数和并发数，才能比较不同引擎的调度质量。"
    "工程上还应当记录每个请求的端到端时延、完成token数、排队效应以及设备内存峰值，"
    "这些指标合在一起才能解释吞吐与延迟之间的取舍。"
)

def build_prompt(tokenizer, n_tokens, idx, salt=0):
    topic = f"资料组{idx % 97 + 1:02d}，批次{idx + 1:03d}。"
    if salt:
        topic += f"防复用{salt:04d}。"
    repeats = n_tokens // 180 + 4
    text = topic + CORPUS * repeats
    ids = tokenizer.encode(text)
    if len(ids) < n_tokens:
        text = text + CORPUS * ((n_tokens - len(ids)) // 100 + 4)
        ids = tokenizer.encode(text)
    ids = ids[:n_tokens]
    return tokenizer.decode(ids)

def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    start_ns = time.time_ns()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        dt = time.perf_counter() - t0
        choice = body["choices"][0]
        msg = choice.get("message", {})
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        return {
            "ok": True,
            "start_ns": start_ns,
            "end_ns": time.time_ns(),
            "latency_s": round(dt, 3),
            "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": body.get("usage", {}).get("completion_tokens"),
            "finish": choice.get("finish_reason"),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_head": text[:40],
        }
    except Exception as exc:
        return {"ok": False, "latency_s": round(time.perf_counter() - t0, 3), "error": repr(exc)}

def run_scenario(url, tokenizer, conc, in_len, out_len, timeout, model_name, prompt_salt=0):
    prompts = [build_prompt(tokenizer, in_len, i, prompt_salt) for i in range(conc)]
    # Warm one request at this input length; output kept tiny to bound cost.
    warm_prompt = build_prompt(tokenizer, min(in_len, 128), 900, prompt_salt)
    warm = post(url, {"model": model_name,
                      "messages": [{"role": "user", "content": warm_prompt}],
                      "max_tokens": 8, "temperature": 0, "ignore_eos": True}, timeout)
    barrier = threading.Barrier(conc)
    def worker(i):
        barrier.wait()
        return post(url, {"model": model_name,
                          "messages": [{"role": "user", "content": prompts[i]}],
                          "max_tokens": out_len, "temperature": 0, "ignore_eos": True}, timeout)
    t_start = time.perf_counter()
    benchmark_start_ns = time.time_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
        results = list(pool.map(worker, range(conc)))
    total = time.perf_counter() - t_start
    benchmark_end_ns = time.time_ns()
    oks = [r for r in results if r.get("ok")]
    lat = sorted(r["latency_s"] for r in oks)
    def pct(p):
        if not lat:
            return None
        k = min(len(lat) - 1, int(round((len(lat) - 1) * p)))
        return round(lat[k], 3)
    summary = {
        "concurrency": conc, "input_tokens": in_len, "output_tokens": out_len,
        "ok": len(oks), "failed": conc - len(oks),
        "wall_s": round(total, 3),
        "benchmark_start_ns": benchmark_start_ns,
        "benchmark_end_ns": benchmark_end_ns,
        "latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "latency_p50_s": pct(0.50), "latency_p99_s": pct(0.99),
        "latency_min_s": round(lat[0], 3) if lat else None,
        "latency_max_s": round(lat[-1], 3) if lat else None,
        "completion_tokens_mean": round(statistics.mean([r["completion_tokens"] for r in oks]), 1) if oks else None,
        "finish_reasons": sorted({str(r.get("finish")) for r in oks}),
        "errors": [r.get("error") for r in results if not r.get("ok")][:2],
        "warmup_s": warm.get("latency_s"),
        "prompt_tokens_first": oks[0].get("prompt_tokens") if oks else None,
    }
    return summary, results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--input", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--model", default="Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp")
    ap.add_argument("--out", default="/home/wuwei/InfiniLM/.tmp-dsv4/latency_results.jsonl")
    ap.add_argument("--prompt-salt", type=int, default=0)
    ap.add_argument("--save-details", action="store_true")
    args = ap.parse_args()
    url = args.url.rstrip("/") + "/v1/chat/completions"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    summary, results = run_scenario(url, tokenizer, args.conc, args.input, args.output, args.timeout, args.model, args.prompt_salt)
    summary["engine"] = args.engine
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    if args.save_details:
        with open(args.out + ".details", "a", encoding="utf-8") as f:
            for r in results:
                r.update({"engine": args.engine, "conc": args.conc,
                          "input": args.input, "output": args.output})
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
