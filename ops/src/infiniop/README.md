# InfiniOP 开发者文档

InfiniOP 是 InfiniCore 下属的统一底层算子框架，为相同算子在不同平台提供统一的 C 语言多段式接口。

## 开发流程

1. 根据算子定义设计算子接口，在 [`InfiniCore文档`](https://github.com/InfiniTensor/InfiniCore-Documentation) 中添加算子文档。提交文档 PR 。

2. 在 `include/infiniop/` 中添加算子头文件，并 include 到 `include/infiniop.h` 中。每个算子暴露的接口包括：创建算子描述、获取工作空间大小、执行算子、销毁算子描述。比如：

    ```c
    #ifndef __INFINIOP_ADD_API_H__
    #define __INFINIOP_ADD_API_H__

    #include "../operator_descriptor.h"

    typedef struct InfiniopDescriptor *infiniopAddDescriptor_t;

    __INFINI_C __export infiniStatus_t infiniopCreateAddDescriptor(infiniopHandle_t handle,
                                                            infiniopAddDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t c,
                                                            infiniopTensorDescriptor_t a,
                                                            infiniopTensorDescriptor_t b);

    __INFINI_C __export infiniStatus_t infiniopGetAddWorkspaceSize(infiniopAddDescriptor_t desc, size_t *size);

    __INFINI_C __export infiniStatus_t infiniopAdd(infiniopAddDescriptor_t desc,
                                            void *workspace,
                                            size_t workspace_size,
                                            void *c,
                                            const void *a,
                                            const void *b,
                                            void *stream);

    __INFINI_C __export infiniStatus_t infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc);

    #endif
    ```

    在任何平台都不需要工作空间的算子也可以不提供获取工作空间大小接口。

3. 在 `src/infiniop/ops/` 中添加算子实现目录，并在目录中创建 `operator.cc` 文件实现头文件中的接口，并根据硬件环境分发至不同平台的核函数。你还可以在目录中创建该算子在全平台通用的代码，比如 `causal_softmax/info.h` 中就包含了对 Causal Softmax 算子在创建算子描述时的一些通用的信息获取和输入输出检查。像逐元素类的算子除了计算内核以外大部分逻辑都是一样的，你可以使用 `src/infiniop/elementwise/` 中的通用代码快速适配算子。

4. 在 `src/infiniop/ops/[op]/[device]/` 中添加平台算子实现。注意复用平台公共代码，比如规约计算（`src/infiniop/reduce/`），开发过程中把未来可复用的代码写在相应公用代码目录里。

    一些 CUDA kernel 可以被多个支持 CUDA 的平台公用，可以考虑在头文件中实现，并在多个源文件中使用。 比如 `mul/cuda/kernel.cuh` 中只有 device 测代码，会被多个支持 CUDA 的平台源代码引用。

5. 算子实现可以成功编译安装后，在 `test/infiniop/` 中添加单测脚本，与 PyTorch 实现进行正确性和性能比较。你可以仿照已有的测试脚本进行开发，以使用各种通用的测试功能。测例应覆盖算子常用类型和形状。测试成功之后可以将测例添加至 `scripts/python_test.py` 一键测试脚本中（这样 Github 自动测试也会包含该算子）。
