import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import safetensors
from pathlib import Path

MODEL_PATH = "/home/shared/models/tinymix-8x1b-chat/"

def get_golden_data(state_dict):
    print("--- Calculating Golden Reference Values for C++ Unit Test ---")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    input_text = "<|user|>\nOnce upon a time</s>\n<|assistant|>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        embeddings = model.model.embed_tokens(input_ids)
        layer0 = model.model.layers[0]
        golden_norm_output = layer0.input_layernorm(embeddings)

        config = model.config
        nh, nkvh, dh, d = config.num_attention_heads, config.num_key_value_heads, config.hidden_size // config.num_attention_heads, config.hidden_size
        
        q_w = state_dict[f"model.layers.0.self_attn.q_proj.weight"]
        k_w = state_dict[f"model.layers.0.self_attn.k_proj.weight"]
        v_w = state_dict[f"model.layers.0.self_attn.v_proj.weight"]
        
        _Q = (q_w.reshape([nh, 2, dh // 2, d]).transpose(1, 2).contiguous())
        _K = (k_w.reshape([nkvh, 2, dh // 2, d]).transpose(1, 2).contiguous())
        _V = v_w.reshape([nkvh, dh // 2, 2, d])
        
        golden_qkv_weight_flat = torch.cat([_Q.flatten(), _K.flatten(), _V.flatten()])
        golden_qkv_weight = golden_qkv_weight_flat.reshape(((nh + 2 * nkvh) * dh, d))
        
        golden_qkv_output = F.linear(golden_norm_output, golden_qkv_weight.T)

    # --- Save tensors to binary files ---
    def save_tensor_to_file(tensor, filename):
        # Save as float32 for easy reading in C++
        tensor.cpu().to(torch.float32).flatten().numpy().tofile(filename)
        print(f"Saved tensor with shape {list(tensor.shape)} to '{filename}'")

    save_tensor_to_file(golden_norm_output, "input_tensor.bin")
    save_tensor_to_file(golden_qkv_weight, "weight_tensor.bin")
    save_tensor_to_file(golden_qkv_output, "expected_output.bin")

if __name__ == "__main__":
    print("Loading model weights...")
    state_dict = {}
    for file in sorted(Path(MODEL_PATH).glob("*.safetensors")):
        with safetensors.safe_open(file, "pt") as data:
            for name in data.keys():
                state_dict[name] = data.get_tensor(name)
    
    get_golden_data(state_dict)
