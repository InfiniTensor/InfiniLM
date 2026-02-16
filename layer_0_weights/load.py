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

def read_I32_pytorch_fixed(filename, shape=None, device='cpu'):
    """
    修复版本：避免使用 torch.from_file
    """
    # 使用 numpy 读取，然后转换为 tensor
    np_data = np.fromfile(filename, dtype=np.int32)
    
    # 转换为 tensor
    tensor_i32 = torch.from_numpy(np_data).to(device)
    
    # # 转换为 BF16
    # tensor_bf16 = tensor_uint16.view(torch.bfloat16)
    
    # # 转换为 FP32
    # tensor_fp32 = tensor_bf16.float()
    
    # 重塑形状
    if shape is not None:
        try:
            tensor_i32 = tensor_i32.reshape(shape)
        except RuntimeError as e:
            print(f"形状重塑错误: {e}")
            print(f"数据元素: {tensor_i32.numel()}, 形状需要: {np.prod(shape)}")
    
    return tensor_i32
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

# attn_out, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/attn_buf_reshape.bin", 
#     shape=(190, 16, 128)
# )

# routing_weights_lm , _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/values_expert.bin", 
#     shape=(190, 8)
# )

# routing_weights_lm , _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/values_expert.bin", 
#     shape=(190, 8)
# )

# routing_weights_lm  = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/swiglu.bin", 
#     shape=(1024)
# )
# final_hidden_states_lm = []
# for i in range(190):
#     token_states , _  = read_bf16_pytorch_fixed(
#     f"/home/featurize/work/My_InfiniLM/layer_0_weights/result_token/expert_out_{i}.bin", 
#     shape=(2048)
#     )
#     final_hidden_states_lm.append(token_states)

# final_hidden_states_lm = torch.stack(final_hidden_states_lm, dim = 0)

final_hidden_states_lm , _ = read_bf16_pytorch_fixed(
    "/home/featurize/work/My_InfiniLM/layer_0_weights/save/global_expert_out_residual.bin",
    shape = (190, 2048)
)

# moe_up_buf, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/moe_up_buf.bin",
#     shape = (1024)
# )

# swiglu_result, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/swiglu_result.bin",
#     shape = (1024)
# )

# down_input, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/down_input.bin",
#     shape = (2048)
# )

# expert_output, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/second_token.bin",
#     shape = (2048)
# )

# expert_weight, _ = read_bf16_pytorch_fixed(
#     "/home/featurize/work/My_InfiniLM/layer_0_weights/save/down_weight.bin",
#     shape = (64, 1024, 2048)
# )






# router_array = []

# for i in range(190):
#     routing_weights_lm_tmp = read_I32_pytorch_fixed(
#     f"/home/featurize/work/My_InfiniLM/layer_0_weights/router/router_{i}.bin", 
#     shape=(8)
#     )
#     router_array.append(routing_weights_lm_tmp)

# router_lm = torch.stack(router_array, dim=0)


# v_viewd_torch = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/v_viewd.pt").squeeze(0).to('cpu')
# routing_weights = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/router_logits.pt")
# attn_out_torch = (softmax_qk @ v_viewd_torch).to('cpu')
# selected_experts = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/swiglu_token_0_expert_0.pt")
# first and last is correct
final_hidden_states = torch.load("/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/tmp/output_12_layer.pt")
print("over")