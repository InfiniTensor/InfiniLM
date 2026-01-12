import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, GenerationConfig
import os
import time

# 加载模型和processor
# 修改为使用Qwen3VLForConditionalGeneration和AutoProcessor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/data/shared/models/Qwen3-VL-2B-Instruct/",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("/data/shared/models/Qwen3-VL-2B-Instruct/", trust_remote_code=True)

# 设置生成配置以确保确定性生成
model.generation_config = GenerationConfig.from_pretrained("/data/shared/models/Qwen3-VL-2B-Instruct/", trust_remote_code=True)
model.generation_config.do_sample = False  # 关闭采样以确保确定性
model.generation_config.max_new_tokens = 200

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
# messages = [
#     {
#         "role":"user",
#         "content":[
#             {
#                 "type":"image",
#                 "url": "/data/users/monitor1379/InfiniLM/010P00002405F02D94-1.jpg"
#             },
#             {
#                 "type":"text",
#                 "text":"Describe this image."
#             }
#         ]
#     }
# ]

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

# for k,v in inputs.items():
#     print(k)
#     print(v.shape)
#     print(v.dtype)
#     print(v)

# 添加时间统计逻辑
start_time = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=200, output_attentions=False, return_dict_in_generate=True)
end_time = time.time()

total_time = end_time - start_time
num_steps = len(generated_ids.sequences[0]) - len(inputs['input_ids'][0])  # 减去输入长度得到生成步骤数
avg_time = (total_time / num_steps) * 1000  # 转换为毫秒

generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids.sequences)
]
output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
print(f"Time per step: {avg_time:.3f}ms")