# infinicore::ops 开发指南

infinicore::ops 模块包含了 InfiniCore 所有 C++ 算子的接口和实现。外部用户可以通过 `include/infinicore/ops/*OPNAME*/*OPNAME*.h` 中定义的 C++ 接口进行算子调用。部分算子会通过 pybind 暴露给 python 前端。

## 开发指南

### 1. 算子定义

创建 `include/infinicore/ops/*OPNAME*/*OPNAME*.h` 头文件，并根据算子名称定义算子的类以及外部计算接口（包括 in-place 和 out-of-place 两种模式），注意算子名称不能重复。

一个算子类主要包含以下部分：

- schema 定义，用于描述算子的输入输出参数形式。
- execute 函数，算子的计算逻辑。
- dispatcher 分发器，用于注册算子在不同设备上的 kernel 实现。一个进程中，一种算子只有一个全局分发器，每种设备上只能同时注册一个 kernel 实现，可以多次注册对之前的实现进行覆盖。详细信息请参考 `include/infinicore/ops/common/dispatcher.hpp`。

示例 `Gemm` 算子的头文件如下：

```c++
#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Gemm {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, float, float);
    static void execute(Tensor c, Tensor a, Tensor b, float alpha, float beta);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor gemm(Tensor a, Tensor b, float alpha = 1.0f, float beta = 0.0f);
void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta);

}
```

### 2. 算子实现

在 `src/infinicore/ops/*OPNAME*/*OPNAME*.cpp` 文件中实现算子的计算逻辑。

- execute 函数，使用算子的分发器，调用对应硬件上的核函数。可以通过 `context::setDevice` 来改变当前运行时的设备种类。
- 计算接口，使用 execute 函数实现算子接口的计算逻辑，包括 in-place 和 out-of-place 两种模式，其中 in-place 模式的接口函数名以 `_` 结尾，将输出接口写入给定的参数中；out-of-place 模式的接口会为输出创建新的 Tensor。

示例 `Gemm` 算子的实现如下：

```c++
#include "infinicore/ops/gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Gemm::schema> &Gemm::dispatcher() {
    static common::OpDispatcher<Gemm::schema> dispatcher_;
    return dispatcher_;
};

void Gemm::execute(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    // 检查张量设备是否一致
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    // 将运行时设备设置为与张量一致。若设备为CPU时，该接口不会进行任何操作
    infinicore::context::setDevice(c->device());
    // 根据张量的设备种类选择 kernel，执行计算
    dispatcher().lookup(c->device().getType())(c, a, b, alpha, beta);
}

Tensor gemm(Tensor a, Tensor b, float alpha, float beta) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    gemm_(c, a, b, alpha, beta);
    return c;
}

void gemm_(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    Gemm::execute(c, a, b, alpha, beta);
}

} 
```

### 3. Kernel 注册

在 `src/infinicore/ops/*OPNAME*/` 目录中添加算子和函数实现，并在算子的分发器中进行注册。你可以选择为单个设备、多个设备、或全部平台注册 kernel 实现（函数指针），你还可以通过使用 `override_existing` 模式覆盖之前的实现。具体信息请参考 `include/infinicore/ops/common/dispatcher.hpp`：

```c++
// 为某个设备注册 kernel 实现
void registerDevice(Device::Type device_type, Fn fn, bool override_existing = true);

// 为多个设备注册 kernel 实现
void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true);

// 为全部平台注册 kernel 实现
void registerAll(Fn fn, bool override_existing = true);

// 查找 kernel 实现
Fn lookup(Device::Type device_type) const;
```

如果你为多个（或全部）设备注册了同一个 kernel 实现，那么你需要自行实现不同设备的分发机制。比如本框架中的 InfiniOP 算子库，其算子接口在不同平台都保持了一致，并根据当前设备类型自动分发，因此在注册时会为所有平台注册同一个计算函数。以 Gemm 算子为例：

```c++
namespace infinicore::op::matmul_impl::infiniop {

// InfiniOP 算子缓存（线程级）
thread_local common::OpCache<size_t, infiniopGemmDescriptor_t> caches(
    100,
    [](infiniopGemmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyGemmDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数
void calculate(Tensor c, Tensor a, Tensor b, float alpha, float beta) {
    // ...
    INFINICORE_CHECK_ERROR(infiniopGemm(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), alpha, beta, context::getStream()));
}

// 在加载 InfiniCore 时为全平台注册 InfiniOP实现
static bool registered = []() {
    Gemm::dispatcher().registerAll(&calculate, false);
    return true;
}();
}
```

你可以仿照上面的例子单独为不同平台实现核函数并注册。请注意在 `xmake/*lua` 中添加对源文件的编译方式，并做好跨平台隔离工作以保证项目在别的平台上也可以正常编译。你可以选择像上面的例子一样，通过 `static bool registered = []() {...}` 方式在加载时注册核函数，但请注意避免加载时为同一个算子重复注册不同核函数的未定义行为。你也可以在程序运行时显式地注册算子。

如果你想通过 InfiniOP 库来实现算子，请参考 [`InfiniOP 开发者文档`](src/infiniop/README.md) 文件。

### 4. Python 接口

通过 pybind11 将 C++ 算子暴露给 Python 前端，需要在 `src/infinicore/pybind11/ops/*OPNAME*/` 目录中添加相应的头文件，并在 `src/infinicore/pybind11/ops.hpp` 中调用。之后你需要在 `python/infinicore/ops/` 目录中为算子添加一个 Python 文件，通过调用你刚才定义的 pybind 接口实现你的 Python 接口，并将 Python 接口通过 `python/infinicore/__init__.py` 暴露给外部。

### 5. Python 测试

在实现了 Python 接口后，你需要在 `/test/infinicore/ops/` 中添加相应的算子测试脚本，并确保测试通过。该目录下的测试使用了统一的测试框架，大部分测试功能已经实现，比如根据形状构建随机张量、自动测试算子的正确性和性能等。

你需要继承 `BaseOperatorTest` 类并实现以下方法：

`get_test_cases()`: 返回测试用例列表

`torch_operator()`: PyTorch 参考实现

`infinicore_operator()`: InfiniCore 算子实现

```python
class OpTest(BaseOperatorTest):
    """Add operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Add")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch add implementation"""
        return torch.add(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore add implementation"""
        return infinicore.add(*args, **kwargs)
```

在测试脚本中你需要为算子测试脚本添加测例。你可以使用统一的格式简洁地准备测试所需数据，如`_TEST_CASES_DATA`。

```python
# Test cases format: (shape, a_strides, b_strides, c_strides)
_TEST_CASES_DATA = [
    # Basic cases
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), None),
    # Strided cases
    ((13, 4), None, None, (10, 1)),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    # 3D cases
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), None),
    # Broadcast cases
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    # Large tensors
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), None),
]
```


对于支持多种精度的算子，你可以指定测试通过的误差范围。

```python
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}
```

通过 `parse_test_cases` 函数来解析测例数据，构建具体的测例。OUT OF PLACE以及各种INPLACE操作，需要作为不同的测例传入`get_test_cases`函数。

```python
# Parse test cases
def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for all operation types.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        c_strides = data[3] if len(data) > 3 else None

        # Check if tensors support in-place operations
        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)
        c_supports_inplace = not is_broadcast(c_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype, name="a")
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype, name="b")
            c_spec = TensorSpec.from_tensor(shape, c_strides, dtype, name="c")

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Add - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (add(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={},
                        output_spec=c_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Add - INPLACE(out)",
                    )
                )

            # Test Case 3: In-place on first input (add(a, b, out=a))
            if a_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},  # Use index 0 for first input
                        output_spec=None,
                        comparison_target=0,  # Compare first input
                        tolerance=tolerance,
                        description=f"Add - INPLACE(a)",
                    )
                )

            # Test Case 4: In-place on second input (add(a, b, out=b))
            if b_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},  # Use index 1 for second input
                        output_spec=None,
                        comparison_target=1,  # Compare second input
                        tolerance=tolerance,
                        description=f"Add - INPLACE(b)",
                    )
                )

    return test_cases

```

运行测试指令检查算子的正确性和性能：

```bash
python test/infinicore/ops/add --nvidia --verbose --bench --debug
```
```bash
python test/infinicore/run.py --ops add matmul --nvidia --verbose --bench
```
