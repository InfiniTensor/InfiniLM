import ml_dtypes
import numpy as np


# 现在可以使用bfloat16
def load_bfloat16_with_ml_dtypes(file_path, shape=None):
    """
    使用ml-dtypes加载bfloat16的.bin文件
    
    Args:
        file_path: .bin文件路径
        shape: 期望的形状
    
    Returns:
        bfloat16数组
    """
    # 直接以bfloat16加载
    data = np.fromfile(file_path, dtype='bfloat16')
    
    print(f"文件: {file_path}")
    print(f"元素数量: {data.size}")
    print(f"数据类型: {data.dtype}")
    
    # 重塑形状
    if shape is not None:
        expected_elements = np.prod(shape)
        if data.size == expected_elements:
            data = data.reshape(shape)
            print(f"重塑为: {shape}")
        else:
            print(f"警告: 数据大小{data.size} != 期望{expected_elements}")
    
    return data

# 使用
data = load_bfloat16_with_ml_dtypes("/home/featurize/work/My_InfiniLM/block1/qkv_buf.bin", shape=(6144, 2048))

# 可以进行运算
print(f"统计: 最小值={data.min()}, 最大值={data.max()}, 均值={data.mean()}")
