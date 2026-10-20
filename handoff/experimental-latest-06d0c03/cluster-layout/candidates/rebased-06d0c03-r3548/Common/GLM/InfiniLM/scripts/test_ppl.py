import math
import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/completions")
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    API_URL = "http://localhost:" + str(args.port) + args.endpoint
    CHUNK_SIZE = args.chunk

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Local tokenizer used for chunking
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    total_neg_log_likelihood = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="Evaluating PPL"):
        text = example["text"].strip()
        if not text:
            continue

        # endcode, chunk and decode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk_tokens = tokens[i : min(i + CHUNK_SIZE, len(tokens))]
            chunk_text = tokenizer.decode(chunk_tokens)

            resp = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "",
                    "prompt": chunk_text,
                    "max_tokens": 0,
                    "temperature": 1.0,
                    "echo": True,
                    "logprobs": 0,
                },
            ).json()

            logprobs = resp["choices"][0]["logprobs"]["token_logprobs"]
            # skip first token's None
            valid_logprobs = [lp for lp in logprobs[1:] if lp is not None]

            total_neg_log_likelihood += -sum(valid_logprobs)
            total_tokens += len(valid_logprobs)

    # ==== Compute final PPL ====
    ppl = math.exp(total_neg_log_likelihood / total_tokens)
    print(f"Perplexity: {ppl:.4f}")
