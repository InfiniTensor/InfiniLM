import torch
import numpy as np

def read_bf16_pytorch_fixed(filename, shape=None, device='cpu'):
    """
    修复版本：避免使用 torch.from_file
    """
    # 使用 numpy 读取，然后转换为 tensor
    np_data = np.fromfile(filename, dtype=np.uint16)
    
    # 转换为 tensor
    tensor_uint16 = torch.from_numpy(np_data).to(device)
    
    # 转换为 BF16
    tensor_bf16 = tensor_uint16.view(torch.bfloat16)
    
    # 转换为 FP32
    tensor_fp32 = tensor_bf16.float()
    
    # 重塑形状
    if shape is not None:
        try:
            tensor_fp32 = tensor_fp32.reshape(shape)
            tensor_bf16 = tensor_bf16.reshape(shape)
        except RuntimeError as e:
            print(f"形状重塑错误: {e}")
            print(f"数据元素: {tensor_fp32.numel()}, 形状需要: {np.prod(shape)}")
    
    return tensor_fp32, tensor_bf16

# 使用修复版本
fp32_data, bf16_data = read_bf16_pytorch_fixed(
    "/home/featurize/work/My_InfiniLM/q_buf_190_16_128_rope.bin", 
    shape=(190, 16, 128)
)
torch_data = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/query_states_1_190_2048_rope.pt")
print(fp32_data.size())