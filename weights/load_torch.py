import torch  
  
def read_bfloat16_bin_torch(file_path, shape, offset_bytes=0):  
    """  
    用 PyTorch 读取 bfloat16 二进制文件并转为 float32 张量。  
    """  
    with open(file_path, "rb") as f:  
        f.seek(offset_bytes)  
        buffer = f.read()  
    # 直接从缓冲区创建 bfloat16 张量，然后转为 float32  
    tensor = torch.frombuffer(buffer, dtype=torch.bfloat16)  
    tensor = tensor.to(torch.float32).view(shape)  
    return tensor  
  
# 示例  
shape = (190, 2048)  
d_path = '/home/featurize/work/My_InfiniLM/weights/out_rms.bin'
tensor = read_bfloat16_bin_torch(d_path, shape, offset_bytes=0)  
print(tensor.shape)
print(tensor)