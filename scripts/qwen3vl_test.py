import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, GenerationConfig
import os
import time

# 加载模型和processor
# 修改为使用Qwen3VLForConditionalGeneration和AutoProcessor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/user/workshop/Qwen3-VL-2B-Instruct/",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("/home/user/workshop/Qwen3-VL-2B-Instruct/", trust_remote_code=True)

# 设置生成配置以确保确定性生成
model.generation_config = GenerationConfig.from_pretrained("/home/user/workshop/Qwen3-VL-2B-Instruct/", trust_remote_code=True)
model.generation_config.do_sample = False  # 关闭采样以确保确定性
model.generation_config.max_new_tokens = 200

# 输入消息 - 结合文本和图像（这里仅保留文本示例）
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "山东最高的山是？"
            }
        ]
    }
]

# 处理输入
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
inputs.pop("token_type_ids", None)

print("Input token IDs:", inputs["input_ids"][0].tolist())
print("Input text:", processor.decode(inputs["input_ids"][0]))

# 获取输入信息用于逐token生成
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 记录开始生成时的总token数
initial_length = input_ids.shape[1]
generated_tokens = []
generation_times = []

# 逐token生成
with torch.no_grad():
    current_input_ids = input_ids
    current_attention_mask = attention_mask
    
    # 获取EOS token ID
    eos_token_id = model.generation_config.eos_token_id
    
    for i in range(model.generation_config.max_new_tokens):
        start_time = time.time()
        
        # 单步生成
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
        )
        
        # 获取下一个token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # 检查是否达到结束条件
        if next_token_id.item() == eos_token_id:
            break
            
        # 记录生成时间
        end_time = time.time()
        generation_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 添加到已生成的token中
        generated_tokens.append(next_token_id.item())
        
        # 更新输入以包含新生成的token
        current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((current_attention_mask.shape[0], 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)], dim=1)

# 计算平均生成时间
if generation_times:
    avg_generation_time = sum(generation_times) / len(generation_times)
    print(f"生成的tokens: {generated_tokens}")
    print(f"生成的文本: {processor.decode(generated_tokens, skip_special_tokens=True)}")
    print(f"生成的token数量: {len(generated_tokens)}")
    print(f"平均生成一个token的时间: {avg_generation_time:.3f} ms")
else:
    print("未生成任何新token")