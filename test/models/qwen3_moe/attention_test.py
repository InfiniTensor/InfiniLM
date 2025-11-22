import os
import time
import sys
import safetensors
import torch
from transformers import AutoConfig
from transformers import DynamicCache
from transformers.models import qwen3_moe

WARMUPS = 10
RUNS = 100
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}

DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--model_path",
        action="store",
        help="The directory of the model to be tested",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cpu test",
    )

    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )

    parser.add_argument(
        "--metax",
        action="store_true",
        help="Run metax test",
    )
    parser.add_argument(
        "--moore",
        action="store_true",
        help="Run moore test",
    )
    parser.add_argument(
        "--iluvatar",
        action="store_true",
        help="Run iluvatar test",
    )
    return parser.parse_args()


def torch_synchronize(_device):
    if _device == "cuda":
        torch.cuda.synchronize()
    elif _device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(_device):
    if _device == "cuda":
        torch.cuda.empty_cache()
    elif _device == "musa":
        torch.musa.empty_cache()


def create_Qwen3attention_torch(dir_path, *, device, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(dir_path)
    config.num_hidden_layers = 1
    config._attn_implementation = "sdpa"

    # --------------------------------------------------------------------------------#
    #                创建只包含 attention的模型
    # --------------------------------------------------------------------------------#
    model = qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention(config, layer_idx=0).to(
        device=device, dtype=dtype
    )
    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if "model.layers.0.self_attn." in key:
                    tensors[key[len("model.layers.0.self_attn.") :]] = f.get_tensor(key)
        break
    model.load_state_dict(tensors)

    # --------------------------------------------------------------------------------#
    #                创建 rotary_emb 类
    # --------------------------------------------------------------------------------#
    rotary_emb = qwen3_moe.modeling_qwen3_moe.Qwen3MoeRotaryEmbedding(
        config, device=device
    )
    return model, rotary_emb


def generate_attention_input_torch(
    model, rotary_emb, testcase, device, dtype=torch.bfloat16
):
    config = model.config
    hidden_size = config.hidden_size  # 2048
    head_dim = config.head_dim  # 128
    num_key_value_heads = config.num_key_value_heads
    bs = 1

    req_list = []
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = torch.rand(
            (bs, seq_lens, hidden_size), device=device, dtype=dtype
        )

        attention_mask = None

        past_key_values = DynamicCache(config=config)
        key_states = torch.rand(
            (bs, num_key_value_heads, past_lens, head_dim), device=device, dtype=dtype
        )
        value_states = torch.rand(
            (bs, num_key_value_heads, past_lens, head_dim), device=device, dtype=dtype
        )
        past_key_values.update(key_states, value_states, 0)

        req = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        req_list.append(req)

    return req_list


def benchmark_Qwen3attention_prefill_torch(
    model, rotary_emb, test_cases, device, dtype=torch.bfloat16
):
    """
    Test Qwen3attention.

    """
    req_list = generate_attention_input_torch(
        model, rotary_emb, test_cases, device, dtype=dtype
    )
    req_out_list = []
    for req in req_list:
        # ----------------------------------------- #
        #         获得每个req的数据
        # ----------------------------------------- #
        hidden_states = req["hidden_states"]
        attention_mask = req["attention_mask"]
        past_key_values = req["past_key_values"]

        # ----------------------------------------- #
        #         计算当前所需的sin_table，sin_table
        # ----------------------------------------- #
        cache_lens = past_key_values.get_seq_length()  # kv cache 现在的长度
        bs, seq_len, _ = hidden_states.shape

        position_ids = torch.arange(
            cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device
        ).reshape((bs, seq_len))

        cos_table, sin_table = rotary_emb(hidden_states, position_ids)
        position_embeddings = (sin_table, cos_table)

        # ----------------------------------------- #
        #            计算一次
        # ----------------------------------------- #
        output_device, _ = model(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        # ----------------------------------------- #
        #            得到结果，存储下来
        # ----------------------------------------- #
        output_host = output_device.to("cpu")
        req_out_list.append(output_host)

    torch_synchronize(device)

    for _ in range(WARMUPS):
        for i, req in enumerate(req_list):
            # ----------------------------------------- #
            #          恢复 kv chche的长度
            # ----------------------------------------- #
            origin_len = test_cases["pastlens"][i]
            req["past_key_values"].crop(origin_len)

        for req in req_list:
            # ----------------------------------------- #
            #         获得每个req的数据
            # ----------------------------------------- #

            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]

            # ----------------------------------------- #
            #         计算当前所需的sin_table，sin_table
            # ----------------------------------------- #
            cache_lens = past_key_values.get_seq_length()  # kv cache 现在的长度
            bs, seq_len, _ = hidden_states.shape

            position_ids = torch.arange(
                cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device
            ).reshape((bs, seq_len))

            cos_table, sin_table = rotary_emb(hidden_states, position_ids)
            position_embeddings = (sin_table, cos_table)

            # ----------------------------------------- #
            #            计算一次
            # ----------------------------------------- #
            output_device, _ = model(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

    time_consuming = 0
    for _ in range(RUNS):
        for i, req in enumerate(req_list):
            # ----------------------------------------- #
            #          恢复 kv chche的长度
            # ----------------------------------------- #
            origin_len = test_cases["pastlens"][i]
            req["past_key_values"].crop(origin_len)

        torch_synchronize(device)
        # ----------------------------------------- #
        #       重要：每个req都按整个batch的起始时间计算
        # ----------------------------------------- #
        start_time = time.time()

        for i, req in enumerate(req_list):
            # ----------------------------------------- #
            #         获得每个req的数据
            # ----------------------------------------- #
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]

            # ----------------------------------------- #
            #         计算当前所需的sin_table，sin_table
            # ----------------------------------------- #
            cache_lens = past_key_values.get_seq_length()  # kv cache 现在的长度
            bs, seq_len, _ = hidden_states.shape

            position_ids = torch.arange(
                cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device
            ).reshape((bs, seq_len))

            cos_table, sin_table = rotary_emb(hidden_states, position_ids)
            position_embeddings = (sin_table, cos_table)

            # ----------------------------------------- #
            #            计算一次
            # ----------------------------------------- #
            output_device, _ = model(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            torch_synchronize(device)
            end_time = time.time()

            # 记录每个req从进入所有req进入推理到自己结束的时间
            time_consuming += end_time - start_time

    out_token_count = RUNS * len(req_list)

    latency = time_consuming * 1000 / out_token_count

    print(
        f"\t WARMUPS={WARMUPS} RUNS={RUNS}, Attention Torch, average TTFT: {round(latency, 2)} ms\n"
    )

    return req_out_list


def benchmark_Qwen3attention_decode_torch(
    model, rotary_emb, test_cases, device, dtype=torch.bfloat16
):
    """
    Test Qwen3attention_decode.
    """
    req_list = generate_attention_input_torch(
        model, rotary_emb, test_cases, device, dtype=dtype
    )
    req_out_list = []
    for req in req_list:
        # ----------------------------------------- #
        #         获得每个req的数据
        # ----------------------------------------- #
        hidden_states = req["hidden_states"]
        attention_mask = req["attention_mask"]
        past_key_values = req["past_key_values"]

        # ----------------------------------------- #
        #         计算当前所需的sin_table，sin_table
        # ----------------------------------------- #
        cache_lens = past_key_values.get_seq_length()  # kv cache 现在的长度
        bs, seq_len, _ = hidden_states.shape

        position_ids = torch.arange(
            cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device
        ).reshape((bs, seq_len))

        cos_table, sin_table = rotary_emb(hidden_states, position_ids)
        position_embeddings = (sin_table, cos_table)

        ##
        output_device, _ = model(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        output_host = output_device.to("cpu")

        req_out_list.append(output_host)

    torch_synchronize(device)

    for req in req_list:
        for _ in range(WARMUPS):
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]

            # ----------------------------------------- #
            #         计算当前所需的sin_table，sin_table
            # ----------------------------------------- #
            cache_lens = past_key_values.get_seq_length()  # kv cache 现在的长度
            bs, seq_len, _ = hidden_states.shape

            position_ids = torch.arange(
                cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device
            ).reshape((bs, seq_len))

            cos_table, sin_table = rotary_emb(hidden_states, position_ids)
            position_embeddings = (sin_table, cos_table)

            # ----------------------------------------- #
            #            计算一次
            # ----------------------------------------- #

            output_device, _ = model(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

    # ----------------------------------------- #
    #          恢复 kv chche的长度
    # ----------------------------------------- #
    for i, req in enumerate(req_list):
        origin_len = test_cases["pastlens"][i]
        req["past_key_values"].crop(origin_len)

    torch_synchronize(device)
    start_time = time.time()

    for i, req in enumerate(req_list):
        for _ in range(RUNS):
            # ----------------------------------------- #
            #         获得每个req的数据
            # ----------------------------------------- #
            hidden_states = req["hidden_states"]
            attention_mask = req["attention_mask"]
            past_key_values = req["past_key_values"]

            # -------------------------------------------------------------- #
            #         计算当前所需的sin_table，sin_table
            # -------------------------------------------------------------- #
            cache_lens = past_key_values.get_seq_length()  # kv cache 现在的长度
            bs, seq_len, _ = hidden_states.shape

            position_ids = torch.arange(
                cache_lens, cache_lens + seq_len, dtype=torch.int64, device=device
            ).reshape((bs, seq_len))

            cos_table, sin_table = rotary_emb(hidden_states, position_ids)
            position_embeddings = (sin_table, cos_table)

            # -------------------------------------------------------------- #
            #            计算一次
            # -------------------------------------------------------------- #
            output_device, _ = model(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

            # -------------------------------------------------------------- #
            #            更新hidden_states, ( DynamicCache的类自动更新)
            # -------------------------------------------------------------- #
            req["hidden_states"] = output_device

    torch_synchronize(device)
    end_time = time.time()

    time_consuming = end_time - start_time
    out_token_count = RUNS * len(req_list)

    throughput = out_token_count / time_consuming

    print(
        f"\t WARMUPS={WARMUPS} RUNS={RUNS}, Attention Torch, average throughput: {round(throughput, 2)} tok/s \n"
    )

    return req_out_list


if __name__ == "__main__":
    args = get_args()
    print(args)

    model_path = args.model_path
    dtype = torch.bfloat16

    # Parse command line arguments
    device = "cpu"
    if args.cpu:
        device = "cpu"
    elif args.nvidia:
        device = "cuda"
    elif args.metax:
        device = "cuda"
    elif args.moore:
        device = "musa"
        import torch_musa
    elif args.iluvatar:
        device = "cuda"
    else:
        print(
            "Usage:  python test/models/qwen3_moe/attention_test.py [--cpu | --nvidia | --metax | --moore | --iluvatar] --model_path=<path/to/model_path>"
        )
        sys.exit(1)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    model, rotary_emb = create_Qwen3attention_torch(
        model_path, device=device, dtype=dtype
    )
    print("\n")
    print("*" * 130)
    print("Test Qwen3attention ")
    print("*" * 130)
    print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
    output_prefill = benchmark_Qwen3attention_prefill_torch(
        model, rotary_emb, PREFILL_TESTCASES, device, dtype=dtype
    )

    print("\n")
    print("-" * 130)
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    output_decode = benchmark_Qwen3attention_decode_torch(
        model, rotary_emb, DECODE_TESTCASES, device, dtype=dtype
    )

    # clean up device memory
    del model
    torch_empty_cache(device)
