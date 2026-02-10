import numpy as np  
  
def read_bfloat16_bin(file_path, shape, offset_bytes=0, dtype=np.float32):  
    """  
    读取 bfloat16 二进制文件并转为 float32 numpy 数组。  
      
    Args:  
        file_path: 二进制文件路径  
        shape: 目标张量形状（tuple/list）  
        offset_bytes: 数据起始偏移量（字节），对应 Tensor::_offset  
        dtype: 输出数组类型，默认 np.float32  
    Returns:  
        numpy.ndarray  
    """  
    # 读取文件  
    with open(file_path, "rb") as f:  
        f.seek(offset_bytes)  
        raw = np.frombuffer(f.read(), dtype=np.uint16)  # 读取为 uint16  
      
    # 将 bfloat16 转为 float32（将高 16 位放入 float32，低 16 位置零）  
    # 等价于 C++ 中的 bf16_to_f32  
    raw_f32 = raw.astype(np.uint32) << 16  
    arr = raw_f32.view(dtype).reshape(shape)  
    return arr  
  
# 示例用法  
shape = (190, 2048)  # 替换为你的张量形状  
offset = 0            # 如果 Tensor 有 _offset，请设置对应字节数  
d_path = '/home/featurize/work/My_InfiniLM/weights/out_rms.bin'
data = read_bfloat16_bin(d_path, shape, offset_bytes=offset)  
print(data)