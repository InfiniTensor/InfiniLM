import os
import sys
import json
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
import numpy as np
import math

# ===============================================================
# Configuration
# ===============================================================
MODEL_PATH = "/home/shared/models/tinymix-8x1b-chat/"

# This is the file you will create from your C++ engine using GDB
CPP_DUMP_FILE_PATH = "./qk_buf.bin" 
# ===============================================================


def get_golden_attention_scores(model, tokenizer, device):
    """
    Performs a forward pass on the HF model to get the "golden"
    attention scores (before softmax) for the first layer.
    """
    print("--- Calculating Golden Reference Values (Hugging Face) ---")
    
    model.to(device)
    model.eval()

    # 1. Prepare input
    input_text = "<|user|>\nOnce upon a time</s>\n<|assistant|>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 2. Perform a full forward pass, explicitly asking for attention weights.
        # The `output_attentions=True` flag tells the model to return the scores.
        outputs = model(input_ids, output_attentions=True)
        
        # The `outputs.attentions` is a tuple containing the attention weights for each layer.
        # We want the weights from the first layer (index 0).
        # Note: These are the scores AFTER the softmax operation.
        attn_weights = outputs.attentions[0]

        print(f"Golden attention score shape: {attn_weights.shape}")
        return attn_weights


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- Load Models and Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ✨ FIX: Force the model to use the "eager" attention implementation.
    # The default "sdpa" (scaled dot-product attention) is optimized and does not
    # support returning attention weights, which would cause `outputs.attentions` to be `None`.
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, # Mixtral uses bfloat16
        attn_implementation="eager" 
    )

    # --- Get the Golden Tensor ---
    golden_scores = get_golden_attention_scores(hf_model, tokenizer, device)

    # --- Load Your C++ Engine's Tensor ---
    print(f"\n--- Loading C++ tensor data from '{CPP_DUMP_FILE_PATH}' ---")
    if not os.path.exists(CPP_DUMP_FILE_PATH):
        print(f"❌ ERROR: Dump file not found.")
        print("Please run your C++ program under GDB and dump the 'qk_buf' memory to this file.")
        return

    # Load raw bytes from the file and interpret them as bfloat16
    with open(CPP_DUMP_FILE_PATH, 'rb') as f:
        raw_bytes = f.read()
    
    your_scores_flat = torch.frombuffer(raw_bytes, dtype=torch.bfloat16)
    
    # Slice the loaded tensor to the expected size before reshaping
    expected_num_elements = golden_scores.numel()
    
    if your_scores_flat.numel() < expected_num_elements:
        print(f"❌ ERROR: Dump file is too small. Expected at least {expected_num_elements} elements, but file only has {your_scores_flat.numel()}.")
        return
        
    your_scores_sliced = your_scores_flat[:expected_num_elements]
    
    try:
        # Note: Your C++ code dumps the scores BEFORE softmax, while the HF API
        # returns them AFTER softmax. We reshape to the same shape for comparison.
        your_scores = your_scores_sliced.reshape(golden_scores.shape)
        print(f"Your C++ tensor shape (after slicing and reshaping): {your_scores.shape}")
    except Exception as e:
        print(f"❌ ERROR: Could not reshape C++ tensor even after slicing.")
        print(e)
        return

    # --- Compare Tensors ---
    print("\n--- Comparing Golden Reference vs. Your C++ Engine ---")
    print("Note: Comparing your pre-softmax scores with the golden post-softmax scores.")
    print("Look for NaN/Inf values or wildly different magnitudes.")
    
    try:
        # We can't use assert_close because the values are different (pre vs post softmax).
        # Instead, we check for basic sanity.
        if torch.isnan(your_scores).any() or torch.isinf(your_scores).any():
             raise ValueError("Your C++ tensor contains NaN or Inf values.")
        
        print("\n" + "="*80)
        print("✅ PARTIAL SUCCESS: Your C++ tensor is sane (no NaN/Inf).")
        print("Golden (post-softmax) first 5 values:", golden_scores.flatten()[:5])
        print("Yours (pre-softmax) first 5 values:  ", your_scores.flatten()[:5])
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ❌ ❌ FAILURE: Your C++ tensor is numerically unstable. ❌ ❌ ❌")
        print("This confirms a mathematical bug in your attention 'blueprint'.")
        print("="*80)
        print("\nError details:")
        print(e)


if __name__ == "__main__":
    main()
