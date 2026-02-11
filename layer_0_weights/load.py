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
q_norm, _ = read_bf16_pytorch_fixed(
    "/home/featurize/work/My_InfiniLM/layer_0_weights/q_buf_190_2048_norm.bin", 
    shape=(190, 2048)
)

reranged, _ = read_bf16_pytorch_fixed(
    "/home/featurize/work/My_InfiniLM/dst.bin", 
    shape=(190, 16, 128)
)

roped, _ = read_bf16_pytorch_fixed(
    "/home/featurize/work/My_InfiniLM/dst_rope.bin", 
    shape=(190, 16, 128)
)
# k_norm, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/k_buf_190_2048_norm.bin", 
#     shape=(190, 2048)
# )

# # 使用修复版本
# q_viewd, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/q_buf_190_16_128_view.bin", 
#     shape=(190, 2048)
# )
# k_viewd, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/k_buf_190_16_128_view.bin", 
#     shape=(190, 2048)
# )

# q_roped, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/q_rope.bin", 
#     shape=(190, 16, 128)
# )

# k_roped, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/k_rope.bin", 
#     shape=(190, 16, 128)
# )

q_norm_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/q_norm.pt").squeeze(0).to('cpu')
# # k_norm_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/k_norm.pt").squeeze(0).to('cpu')
q_viewd_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/q_viewd.pt").squeeze(0).to('cpu')
# # k_viewd_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/q_viewd.pt")

q_roped_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/q_rope.pt")

print("over")