import numpy as np
data_path = '/home/featurize/work/My_InfiniLM/attention/q_weight_buf.bin'
def convert_bf16_to_f32(bf16_array):  
    """将 BF16 numpy 数组转换为 FP32"""  
    bf16_uint16 = bf16_array.astype(np.uint16)  
    bf16_uint32 = bf16_uint16.astype(np.uint32) << 16  
    return bf16_uint32.view(np.float32)


data = np.fromfile(data_path, dtype=np.uint16) 
data_f32 = convert_bf16_to_f32(data)
print(data_f32)