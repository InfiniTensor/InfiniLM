#!/usr/bin/env python3
"""
最简单的transformers模型推理脚本
用于快速测试模型输出
"""

import torch
import argparse
import warnings
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 过滤 transformers 的警告
warnings.filterwarnings("ignore")
# 设置环境变量以抑制 generation flags 警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


def main():
    parser = argparse.ArgumentParser(description='简单的模型推理脚本')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--prompt', type=str, default='你好，请介绍一下你自己。',
                       help='输入提示词')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='最大生成token数')
    parser.add_argument('--device', type=str, default=None,
                       help='指定设备，如: cuda:0 (默认自动选择)')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'float32'],
                       help='数据类型')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    print(f"加载模型: {args.model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype_map[args.dtype],
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    
    # 编码输入
    print(f"\n输入提示: {args.prompt}")
    encoded = tokenizer(args.prompt, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    input_length = input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id
    
    # 预热（可选）
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 生成回答并测量时间
    print("\n生成中...")
    with torch.no_grad():
        # Prefill 阶段：处理输入 prompt
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.time()
        
        # 第一次前向传播（prefill）
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        
        # Decode 阶段：生成新 tokens
        generated_ids = [input_ids, next_token_id]
        decode_times = []
        
        for _ in range(args.max_new_tokens - 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_start = time.time()
            
            # 使用 past_key_values 进行增量生成
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            generated_ids.append(next_token_id)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_end = time.time()
            decode_times.append(decode_end - decode_start)
            
            # 检查是否生成结束符
            if next_token_id.item() == tokenizer.eos_token_id:
                break
        
        # 合并所有生成的 tokens
        outputs = torch.cat(generated_ids, dim=1)
    
    # 计算统计信息
    decode_time_total = sum(decode_times)
    decode_count = len(decode_times)
    generated_length = outputs.shape[1] - input_length
    
    prefill_throughput = input_length / prefill_time if prefill_time > 0 else 0
    decode_throughput = decode_count / decode_time_total if decode_time_total > 0 else 0
    avg_decode_time = decode_time_total / decode_count if decode_count > 0 else 0
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 打印结果
    print("\n" + "="*80)
    print("模型回答:")
    print("="*80)
    print(response)
    print("="*80)
    print("\n性能统计:")
    print("="*80)
    print(f"输入长度: {input_length} tokens")
    print(f"生成长度: {generated_length} tokens")
    print(f"Prefill 时间: {prefill_time*1000:.2f} ms")
    print(f"Prefill 吞吐率: {prefill_throughput:.2f} tokens/s")
    print(f"Decode 总时间: {decode_time_total*1000:.2f} ms")
    print(f"Decode 平均时间: {avg_decode_time*1000:.2f} ms/token")
    print(f"Decode 吞吐率: {decode_throughput:.2f} tokens/s")
    print(f"总时间: {(prefill_time + decode_time_total)*1000:.2f} ms")
    print("="*80)


if __name__ == "__main__":
    main()
