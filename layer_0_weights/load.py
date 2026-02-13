import torch
import numpy as np
import torch.nn.functional as F
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

def read_f32_pytorch_fixed(filename, shape=None, device='cpu'):
    """
    修复版本：避免使用 torch.from_file
    """
    # 使用 numpy 读取，然后转换为 tensor
    np_data = np.fromfile(filename, dtype=np.float32)
    
    # 转换为 tensor
    tensor_f32 = torch.from_numpy(np_data).to(device)
    
    # # 转换为 BF16
    # tensor_bf16 = tensor_uint16.view(torch.bfloat16)
    
    # # 转换为 FP32
    # tensor_fp32 = tensor_bf16.float()
    
    # 重塑形状
    if shape is not None:
        try:
            tensor_fp32 = tensor_f32.reshape(shape)
        except RuntimeError as e:
            print(f"形状重塑错误: {e}")
            print(f"数据元素: {tensor_f32.numel()}, 形状需要: {np.prod(shape)}")
    
    return tensor_f32
# gemm, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/qk_gemm_softmax.bin", 
#     shape=(16, 190, 190)
# )

# v_viewd, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/v_viewd.bin", 
#     shape=(190, 16, 128)
# )

# v_permute, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/v_permute_rerange.bin", 
#     shape=(16, 190, 128)
# )

attn_out, _ = read_bf16_pytorch_fixed(
    "/home/featurize/work/My_InfiniLM/attn_buf_reshape.bin", 
    shape=(190, 16, 128)
)


# v_viewd_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/v_viewd.pt").squeeze(0).to('cpu')
softmax_qk = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/attn_output_190_2048.pt")
# attn_out_torch = (softmax_qk @ v_viewd_torch).to('cpu')

# first and last is correct

print("over")