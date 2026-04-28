#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "rearrange_kernel.cuh"
#include "rearrange_nvidia.cuh"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdint.h>
#include <vector>

namespace op::rearrange::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();

    CHECK_OR_RETURN(x_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() == ndim, INFINI_STATUS_BAD_TENSOR_SHAPE);
    // 保存临时vector对象
    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    auto y_strides = y_desc->strides();
    auto x_strides = x_desc->strides();

    CHECK_SAME_SHAPE(x_shape, y_shape);

    auto meta = utils::RearrangeMeta::create(
        y_shape.data(),
        y_strides.data(),
        x_strides.data(),
        ndim,
        infiniSizeOf(dtype));

    CHECK_RESULT(meta);

    *desc_ptr = new Descriptor(
        std::move(*meta),
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// 维度信息结构
struct Dim {
    size_t len;
    ARRAY_TYPE_STRIDE src_stride;
    ARRAY_TYPE_STRIDE dst_stride;
};

// 分割维度结构
struct SplitDim {
    size_t choose_idx;
    size_t num_per_block;
    size_t num_per_grid;
    int array_struct_idx_block;
    int array_struct_idx_grid;
    size_t dim_len;
};

/**
 * 根据给定的元数据准备张量重排参数，该函数主要完成以下工作：
 * 1. 根据原始元数据调整单元大小，获取更适合GPU处理的单元大小
 * 2. 将维度分配为CUDA块（block）维度和网格（grid）维度：
 *    该步骤是核心，目标是为每个block分配尽可能多的相对连续的数据进行处理，
 *    对无法完整放入块的维度进行分割，并记录分割维度信息，用于防止kernel访问越界，最大化内存访问局部性和计算效率
 */
utils::Result<RearrangeParams> prepareRearrangeParams(const utils::RearrangeMeta &original_meta, int max_threads) {
    RearrangeParams params;

    // 获取更适合GPU处理的单元大小，这里使用2的幂次方
    auto meta_result = original_meta.distributeUnit({32, 16, 8, 4, 2, 1});
    CHECK_RESULT(meta_result);
    const utils::RearrangeMeta &meta = meta_result.take();

    // 获取维度信息
    const size_t ndim = meta.ndim();
    const size_t unit = meta.unit();

    // 特殊情况：无维度，只需要简单复制
    if (ndim == 0) {
        params.block_dim = 0;
        params.block_len_total = 1;
        params.block_len = {static_cast<ARRAY_TYPE_SIZE>(1)};
        params.src_block_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.dst_block_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.grid_len = {static_cast<ARRAY_TYPE_SIZE>(1)};
        params.src_grid_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.dst_grid_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.unit_size = unit;
        return utils::Result<RearrangeParams>(params);
    }

    // 从元数据中提取必要的信息
    const ptrdiff_t *idx_strides = meta.idx_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    const ptrdiff_t *src_strides = meta.src_strides();

    // 准备维度信息
    std::vector<Dim> dims;
    std::vector<size_t> shape;
    dims.reserve(ndim);
    shape.reserve(ndim);

    auto prev_idx_stride = meta.count();
    for (size_t i = 0; i < ndim; ++i) {
        size_t len = prev_idx_stride / idx_strides[i];
        shape.push_back(len);
        dims.push_back({len, src_strides[i], dst_strides[i]});
        prev_idx_stride = idx_strides[i];
    }

    std::vector<bool> block_dim_choose(ndim, false);
    std::vector<SplitDim> split_dims;

    // 初始化计数器
    size_t block_elements = 1;

    std::vector<size_t> dim_order(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        dim_order[i] = i;
    }

    // 按src_stride升序排序，贪心选择
    std::sort(dim_order.begin(), dim_order.end(),
              [&dims](size_t a, size_t b) {
                  return std::abs(dims[a].src_stride) < std::abs(dims[b].src_stride);
              });

    // 维度选择循环
    for (size_t i = 0; i < ndim; ++i) {
        size_t dim_idx = dim_order[i];
        size_t dim_len = shape[dim_idx];

        if (block_elements * dim_len <= (size_t)max_threads) {
            block_dim_choose[dim_idx] = true;
            block_elements *= dim_len;
        } else if (block_elements > 1 && dim_len > 1) {
            // 需要分割此维度
            size_t num_per_block = std::min(dim_len, (size_t)max_threads / block_elements);
            if (num_per_block > 0) {
                size_t num_per_grid = (dim_len + num_per_block - 1) / num_per_block;

                SplitDim split_dim = {
                    dim_idx,       // choose_idx
                    num_per_block, // num_per_block
                    num_per_grid,  // num_per_grid
                    0,             // array_struct_idx_block (待更新)
                    0,             // array_struct_idx_grid (待更新)
                    dim_len        // original dimension length
                };
                split_dims.push_back(split_dim);
                block_elements *= num_per_block;
            }
            break;
        }
    }

    if (block_elements == 1 && ndim > 0) {
        size_t dim_idx = dim_order[0];
        size_t dim_len = shape[dim_idx];

        if (dim_len <= (size_t)max_threads) {
            block_dim_choose[dim_idx] = true;
            block_elements = dim_len;
        } else {
            // 需要分割
            size_t num_per_block = std::min(dim_len, (size_t)max_threads);
            size_t num_per_grid = (dim_len + num_per_block - 1) / num_per_block;

            SplitDim split_dim = {
                dim_idx,
                num_per_block,
                num_per_grid,
                0,
                0,
                dim_len};
            split_dims.push_back(split_dim);
            block_elements = num_per_block;
        }
    }

    // 准备block维度相关参数
    size_t block_dim = 0;
    size_t block_len_total = block_elements;

    std::vector<ARRAY_TYPE_SIZE> block_len;
    std::vector<ARRAY_TYPE_STRIDE> src_block_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_block_stride;

    std::vector<ARRAY_TYPE_SIZE> grid_len;
    std::vector<ARRAY_TYPE_STRIDE> src_grid_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_grid_stride;

    // 处理block维度，填充block_len和block_stride
    for (size_t i = 0; i < ndim; ++i) {
        if (block_dim_choose[i]) {
            block_len.push_back(shape[i]);
            src_block_stride.push_back(dims[i].src_stride);
            dst_block_stride.push_back(dims[i].dst_stride);
            block_dim += 1;
        }

        // 处理分割维度的block部分
        for (size_t j = 0; j < split_dims.size(); ++j) {
            if (i == split_dims[j].choose_idx) {
                block_len.push_back(split_dims[j].num_per_block);
                src_block_stride.push_back(dims[i].src_stride);
                dst_block_stride.push_back(dims[i].dst_stride);
                split_dims[j].array_struct_idx_block = static_cast<int>(block_dim);
                block_dim += 1;
            }
        }
    }

    // 处理grid维度，填充grid_len和grid_stride
    for (size_t i = 0; i < ndim; ++i) {
        if (!block_dim_choose[i]) {
            bool is_split = false;

            // 检查是否是分割维度
            for (size_t j = 0; j < split_dims.size(); ++j) {
                if (i == split_dims[j].choose_idx) {
                    is_split = true;
                    grid_len.push_back(split_dims[j].num_per_grid);
                    src_grid_stride.push_back(dims[i].src_stride * split_dims[j].num_per_block);
                    dst_grid_stride.push_back(dims[i].dst_stride * split_dims[j].num_per_block);
                    split_dims[j].array_struct_idx_grid = static_cast<int>(grid_len.size() - 1);
                    break;
                }
            }

            // 如果不是分割维度，则作为完整的grid维度
            if (!is_split) {
                grid_len.push_back(shape[i]);
                src_grid_stride.push_back(dims[i].src_stride);
                dst_grid_stride.push_back(dims[i].dst_stride);
            }
        }
    }

    // 如果grid_len为空，添加一个默认值
    if (grid_len.empty()) {
        grid_len.push_back(1);
        src_grid_stride.push_back(0);
        dst_grid_stride.push_back(0);
    }

    // 处理约束条件 - 使用与Rust版本相似的逻辑
    std::vector<Constraint<ARRAY_TYPE_SIZE>> constraints;

    // 限制最多处理2个约束条件
    for (size_t i = 0; i < split_dims.size(); ++i) {
        if (split_dims[i].dim_len % split_dims[i].num_per_block == 0) {
            continue;
        }
        Constraint<ARRAY_TYPE_SIZE> constraint;
        constraint.grid_idx = split_dims[i].array_struct_idx_grid;
        constraint.block_idx = split_dims[i].array_struct_idx_block;
        constraint.grid_div_block = split_dims[i].num_per_block;
        constraint.total_len = split_dims[i].dim_len;
        constraints.push_back(constraint);

        if (constraints.size() >= 2) {
            break;
        }
    }

    // 设置参数
    params.block_dim = block_dim;
    params.block_len_total = block_len_total;
    params.block_len = block_len;
    params.src_block_stride = src_block_stride;
    params.dst_block_stride = dst_block_stride;
    params.grid_len = grid_len;
    params.src_grid_stride = src_grid_stride;
    params.dst_grid_stride = dst_grid_stride;
    params.constraints = constraints;
    params.unit_size = unit;

    return utils::Result<RearrangeParams>(params);
}

// 带约束的内核启动模板函数
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    void *y,
    const void *x,
    size_t grid_size,
    const RearrangeParams &params,
    size_t unit_size,
    cudaStream_t stream) {

    // 获取内核函数
    RearrangeParams params_copy = params; // 创建一个非const副本
    auto kernel_func_result = getRearrangeKernel(params_copy);

    CHECK_RESULT(kernel_func_result);
    auto kernel_func = kernel_func_result.take();

    // 创建非const的临时变量
    size_t block_dim = params.block_dim;
    size_t block_len_total = params.block_len_total;

    // 检查向量尺寸是否合理
    if (params.block_len.size() < block_dim || params.src_block_stride.size() < block_dim || params.dst_block_stride.size() < block_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (params.grid_len.empty() || params.src_grid_stride.empty() || params.dst_grid_stride.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const Constraint<ARRAY_TYPE_SIZE> *constraints_data;
    auto empty_constraints = Constraint<ARRAY_TYPE_SIZE>();
    if (params.constraints.empty()) {
        constraints_data = &empty_constraints;
    } else {
        constraints_data = params.constraints.data();
    }

    void *args[]
        = {
            &y, &x,
            &block_dim,
            &block_len_total,
            const_cast<void *>(static_cast<const void *>(params.block_len.data())),
            const_cast<void *>(static_cast<const void *>(params.src_block_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.dst_block_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.grid_len.data())),
            const_cast<void *>(static_cast<const void *>(params.src_grid_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.dst_grid_stride.data())),
            const_cast<void *>(static_cast<const void *>(constraints_data))};

    CHECK_CUDA(cudaLaunchKernel(
        kernel_func,
        static_cast<unsigned int>(grid_size), static_cast<unsigned int>(BLOCK_SIZE),
        args, 0, stream));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 如果没有维度，直接进行内存拷贝
    if (_meta.ndim() == 0) {
        CHECK_OR_RETURN(cudaMemcpyAsync(y, x, _meta.unit(), cudaMemcpyDeviceToDevice, cuda_stream) == cudaSuccess,
                        INFINI_STATUS_INTERNAL_ERROR);
        return INFINI_STATUS_SUCCESS;
    }

    // 获取设备属性
    int max_threads = _opaque->internal->maxThreadsPerBlock();

    // 准备参数
    auto params_result = prepareRearrangeParams(_meta, std::min(CUDA_BLOCK_SIZE_1024, max_threads));
    CHECK_RESULT(params_result);
    auto params = params_result.take();

    // 计算grid大小
    size_t grid_size = 1;
    for (size_t i = 0; i < params.grid_len.size(); ++i) {
        grid_size *= params.grid_len[i];
    }

    // 检查grid大小是否为0
    if (grid_size == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 根据设备属性选择合适的内核
    infiniStatus_t status = INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    size_t block_size = params.block_len_total;

    if (block_size <= CUDA_BLOCK_SIZE_512) {
        status = launchKernel<CUDA_BLOCK_SIZE_512>(y, x, grid_size, params, _meta.unit(), cuda_stream);
    } else if (block_size <= CUDA_BLOCK_SIZE_1024) {
        status = launchKernel<CUDA_BLOCK_SIZE_1024>(y, x, grid_size, params, _meta.unit(), cuda_stream);
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return status;
}

} // namespace op::rearrange::nvidia
