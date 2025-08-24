import torch
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def row_major_strides(shape):
    """生成张量的行优先stride
    
    Args:
        shape: 张量形状
    
    Returns:
        行优先strides列表
    """
    stride = 1
    strides = [1]
    for dim in reversed(shape[1:]):
        stride *= dim
        strides.insert(0, stride)
    return strides

def column_major_strides(shape):
    """生成张量的列优先stride
    
    Args:
        shape: 张量形状
    
    Returns:
        列优先strides列表
    """
    stride = 1
    strides = [stride]
    for dim in shape[:-1]:
        stride *= dim
        strides.append(stride)
    return strides

def rearrange_using_torch(src: torch.Tensor, dst_strides: List[int]) -> torch.Tensor:
    """
    使用torch的rearrange函数计算结果
    Args:
        src: 源张量 (torch.Tensor)
        dst_strides: 目标张量的strides
    Returns:
        重排后的张量 (torch.Tensor)
    """
    # 直接调用rearrange_torch
    y_ = src.clone()
    y_.set_(y_.untyped_storage(), 0, src.shape, dst_strides)
    y_[:] = src.view_as(y_)
    return y_


class RearrangeTestCase(InfiniopTestCase):
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        shape: List[int] | None,
        src_strides: List[int] | None,
        dst_strides: List[int] | None,
    ):
        super().__init__("rearrange")
        self.src = src
        self.dst = dst
        self.shape = shape
        self.src_strides = src_strides
        self.dst_strides = dst_strides
        
    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # 写入形状信息
        if self.shape is not None:
            test_writer.add_array(test_writer.gguf_key("src.shape"), self.shape)
            test_writer.add_array(test_writer.gguf_key("dst.shape"), self.shape)
        
        # 写入strides信息
        if self.src_strides is not None:
            test_writer.add_array(test_writer.gguf_key("src.strides"), gguf_strides(*self.src_strides))
        test_writer.add_array(
            test_writer.gguf_key("dst.strides"),
            gguf_strides(*self.dst_strides if self.dst_strides is not None else contiguous_gguf_strides(self.shape))
        )
        
        # 转换torch tensor为numpy用于写入文件
        src_numpy = self.src.detach().cpu().numpy()
        dst_numpy = self.dst.detach().cpu().numpy()
        
        # 写入张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("src"),
            src_numpy,
            raw_dtype=np_dtype_to_ggml(src_numpy.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("dst"),
            dst_numpy,
            raw_dtype=np_dtype_to_ggml(dst_numpy.dtype),
        )
        
        # 计算并写入答案
        dst_strides_for_ans = self.dst_strides if self.dst_strides is not None else list(contiguous_gguf_strides(self.shape))
        ans_torch = rearrange_using_torch(self.src, dst_strides_for_ans)
        ans_numpy = ans_torch.detach().cpu().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(src_numpy.dtype),
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rearrange.gguf")
    test_cases = []

    _TEST_CASES_ = [
        # (shape, src_stride, dst_stride)
        ((100, 100), (1, 100), (100, 1)),
        ((4, 4), (1, 4), (4, 1)),
        ((4, 6, 64), (64, 4*64, 1), (6*64, 64, 1)),
        ((2000, 2000), (1, 2000), (2000, 1)),
        ((2001, 2001), (1, 2001), (2001, 1)),
        ((2, 2, 2, 4), (16, 8, 4, 1), (16, 8, 1, 2)),
        ((3, 4, 7, 53, 9), row_major_strides((3, 4, 7, 53, 9)), column_major_strides((3, 4, 7, 53, 9))),
        ((3, 4, 50, 50, 5, 7), row_major_strides((3, 4, 50, 50, 5, 7)), column_major_strides((3, 4, 50, 50, 5, 7))),
    ]

    _TENSOR_DTYPES_ = [torch.float32, torch.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, src_strides, dst_strides in _TEST_CASES_:
            # 生成源张量，使用torch
            src = torch.rand(*shape, dtype=dtype)
            # 生成目标张量，使用正确的形状
            dst = torch.empty(shape, dtype=dtype)
            
            test_case = RearrangeTestCase(
                src=src,
                dst=dst,
                shape=shape,
                src_strides=src_strides,
                dst_strides=dst_strides,
            )
            test_cases.append(test_case)        

    test_writer.add_tests(test_cases)
    test_writer.save() 
