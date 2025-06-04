#ifndef __REARRANGE_MACA_KERNEL_H__
#define __REARRANGE_MACA_KERNEL_H__

#include "../../../devices/maca/common_maca.h"
#include "../../../devices/maca/maca_kernel_common.h"

#define ARRAY_TYPE_STRIDE ptrdiff_t
#define ARRAY_TYPE_SIZE size_t

// 与 DEFINE_KERNELS_BY_CONSTRAINT 耦合，需要同时修改
#define MAX_BLOCK_ARRAY_SIZE 5
#define MAX_GRID_ARRAY_SIZE 5

template <int ArrSize, typename ArrayType>
struct ArrayStruct {
    ArrayType a[ArrSize];
};

// 各个元素分别代表：[grid_idx, block_idx, grid的stride相对于block的倍数，总的len限制]
template <typename ElementType>
struct Constraint {
    ElementType grid_idx;
    ElementType block_idx;
    ElementType grid_div_block;
    ElementType total_len;
};

#define IF_CONSTRAINT_0 , const ArrayStruct<1, Constraint<ARRAY_TYPE_SIZE>> constraints
#define IF_CONSTRAINT_1 , const ArrayStruct<1, Constraint<ARRAY_TYPE_SIZE>> constraints
#define IF_CONSTRAINT_2 , const ArrayStruct<2, Constraint<ARRAY_TYPE_SIZE>> constraints

// 定义宏生成内核函数
#define DEFINE_REARRANGE_KERNEL(Tmem_type, constraint_num, block_array_size, grid_array_size)                                                                                    \
    extern "C" __global__ void rearrange_unit_##Tmem_type##_block_##block_array_size##_grid_##grid_array_size##_constrain_##constraint_num(                                      \
        void *__restrict__ dst,                                                                                                                                                  \
        const void *__restrict__ src,                                                                                                                                            \
        const size_t block_dim,                                                                                                                                                  \
        const size_t block_len_total,                                                                                                                                            \
        const ArrayStruct<block_array_size, ARRAY_TYPE_SIZE> block_len,                                                                                                          \
        const ArrayStruct<block_array_size, ARRAY_TYPE_STRIDE> src_block_stride, /* 字节单位的步长 */                                                                     \
        const ArrayStruct<block_array_size, ARRAY_TYPE_STRIDE> dst_block_stride, /* 字节单位的步长 */                                                                     \
        const ArrayStruct<grid_array_size, ARRAY_TYPE_SIZE> grid_len,                                                                                                            \
        const ArrayStruct<grid_array_size, ARRAY_TYPE_STRIDE> src_grid_stride, /* 字节单位的步长 */                                                                       \
        const ArrayStruct<grid_array_size, ARRAY_TYPE_STRIDE> dst_grid_stride  /* 字节单位的步长 */                                                                       \
            IF_CONSTRAINT_##constraint_num) {                                                                                                                                    \
        size_t remaining = threadIdx.x;                                                                                                                                          \
        if (remaining >= block_len_total) {                                                                                                                                      \
            return;                                                                                                                                                              \
        }                                                                                                                                                                        \
                                                                                                                                                                                 \
        /* 声明共享内存 */                                                                                                                                                 \
        __shared__ ptrdiff_t shared_src_offset;                                                                                                                                  \
        __shared__ ptrdiff_t shared_dst_offset;                                                                                                                                  \
                                                                                                                                                                                 \
        if (constraint_num > 0) {                                                                                                                                                \
            __shared__ ARRAY_TYPE_SIZE shared_constraints_grid_idx_multiple[constraint_num > 0 ? constraint_num : 1];                                                            \
                                                                                                                                                                                 \
            if (threadIdx.x == 0) { /* 只让0号线程计算 */                                                                                                                 \
                /* 计算当前block处理的数据在src和dst中的基础偏移(bytes) */                                                                                      \
                ptrdiff_t src_offset = 0;                                                                                                                                        \
                ptrdiff_t dst_offset = 0;                                                                                                                                        \
                ARRAY_TYPE_SIZE constraints_grid_idx_multiple[constraint_num > 0 ? constraint_num : 1];                                                                          \
                                                                                                                                                                                 \
                size_t remaining                                                                                                                                                 \
                    = blockIdx.x;                                                                                                                                                \
                                                                                                                                                                                 \
                for (ssize_t i = grid_array_size - 1; i >= 0; i--) {                                                                                                             \
                    size_t idx = remaining % grid_len.a[i];                                                                                                                      \
                    remaining /= grid_len.a[i];                                                                                                                                  \
                    src_offset += idx * src_grid_stride.a[i];                                                                                                                    \
                    dst_offset += idx * dst_grid_stride.a[i];                                                                                                                    \
                    if (constraint_num > 0) {                                                                                                                                    \
                        for (ssize_t j = 0; j < constraint_num; j++) {                                                                                                           \
                            if (i == constraints.a[j].grid_idx) {                                                                                                                \
                                constraints_grid_idx_multiple[j] = idx * constraints.a[j].grid_div_block;                                                                        \
                            }                                                                                                                                                    \
                        }                                                                                                                                                        \
                    }                                                                                                                                                            \
                }                                                                                                                                                                \
                                                                                                                                                                                 \
                /* 将结果存入共享内存 */                                                                                                                                \
                shared_src_offset = src_offset;                                                                                                                                  \
                shared_dst_offset = dst_offset;                                                                                                                                  \
                for (ssize_t j = 0; j < constraint_num; j++) {                                                                                                                   \
                    shared_constraints_grid_idx_multiple[j] = constraints_grid_idx_multiple[j];                                                                                  \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            /* 确保所有线程都能看到共享内存中的值 */                                                                                                            \
            __syncthreads();                                                                                                                                                     \
                                                                                                                                                                                 \
            /* 所有线程直接使用计算好的偏移值 */                                                                                                                  \
            ptrdiff_t src_offset = shared_src_offset;                                                                                                                            \
            ptrdiff_t dst_offset = shared_dst_offset;                                                                                                                            \
            ARRAY_TYPE_SIZE constraints_grid_idx_multiple[constraint_num > 0 ? constraint_num : 1];                                                                              \
            for (ssize_t j = 0; j < constraint_num; j++) {                                                                                                                       \
                constraints_grid_idx_multiple[j] = shared_constraints_grid_idx_multiple[j];                                                                                      \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            for (ssize_t i = block_array_size - 1; i >= 0; i--) {                                                                                                                \
                size_t idx = remaining % block_len.a[i];                                                                                                                         \
                remaining /= block_len.a[i];                                                                                                                                     \
                /* 计算偏移量 */                                                                                                                                            \
                src_offset += idx * src_block_stride.a[i];                                                                                                                       \
                dst_offset += idx * dst_block_stride.a[i];                                                                                                                       \
                if (constraint_num > 0) {                                                                                                                                        \
                    for (ssize_t j = 0; j < constraint_num; j++) {                                                                                                               \
                        if (i == constraints.a[j].block_idx) {                                                                                                                   \
                            if (constraints_grid_idx_multiple[j] + idx >= constraints.a[j].total_len) {                                                                          \
                                return;                                                                                                                                          \
                            }                                                                                                                                                    \
                        }                                                                                                                                                        \
                    }                                                                                                                                                            \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            src_offset += remaining * src_block_stride.a[0];                                                                                                                     \
            dst_offset += remaining * dst_block_stride.a[0];                                                                                                                     \
            for (ssize_t j = 0; j < constraint_num; j++) {                                                                                                                       \
                if (0 == constraints.a[j].block_idx) {                                                                                                                           \
                    if (constraints_grid_idx_multiple[j] + remaining >= constraints.a[j].total_len) {                                                                            \
                        return;                                                                                                                                                  \
                    }                                                                                                                                                            \
                }                                                                                                                                                                \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            /* 执行数据拷贝，注意offset已经是字节偏移 */                                                                                                         \
            *reinterpret_cast<Tmem_type *>(reinterpret_cast<char *>(dst) + dst_offset) = *reinterpret_cast<const Tmem_type *>(reinterpret_cast<const char *>(src) + src_offset); \
                                                                                                                                                                                 \
        } else {                                                                                                                                                                 \
            if (threadIdx.x == 0) { /* 只让0号线程计算 */                                                                                                                 \
                /* 计算当前block处理的数据在src和dst中的基础偏移(bytes) */                                                                                      \
                ptrdiff_t src_offset = 0;                                                                                                                                        \
                ptrdiff_t dst_offset = 0;                                                                                                                                        \
                size_t remaining = blockIdx.x;                                                                                                                                   \
                                                                                                                                                                                 \
                for (ssize_t i = grid_array_size - 1; i >= 0; i--) {                                                                                                             \
                    size_t idx = remaining % grid_len.a[i];                                                                                                                      \
                    remaining /= grid_len.a[i];                                                                                                                                  \
                    src_offset += idx * src_grid_stride.a[i];                                                                                                                    \
                    dst_offset += idx * dst_grid_stride.a[i];                                                                                                                    \
                }                                                                                                                                                                \
                                                                                                                                                                                 \
                /* 将结果存入共享内存 */                                                                                                                                \
                shared_src_offset = src_offset;                                                                                                                                  \
                shared_dst_offset = dst_offset;                                                                                                                                  \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            /* 确保所有线程都能看到共享内存中的值 */                                                                                                            \
            __syncthreads();                                                                                                                                                     \
                                                                                                                                                                                 \
            /* 所有线程直接使用计算好的偏移值 */                                                                                                                  \
            ptrdiff_t src_offset = shared_src_offset;                                                                                                                            \
            ptrdiff_t dst_offset = shared_dst_offset;                                                                                                                            \
                                                                                                                                                                                 \
            for (ssize_t i = block_array_size - 1; i > 0; i--) {                                                                                                                 \
                size_t idx = remaining % block_len.a[i];                                                                                                                         \
                remaining /= block_len.a[i];                                                                                                                                     \
                /* 计算偏移量 */                                                                                                                                            \
                src_offset += idx * src_block_stride.a[i];                                                                                                                       \
                dst_offset += idx * dst_block_stride.a[i];                                                                                                                       \
            }                                                                                                                                                                    \
                                                                                                                                                                                 \
            src_offset += remaining * src_block_stride.a[0];                                                                                                                     \
            dst_offset += remaining * dst_block_stride.a[0];                                                                                                                     \
                                                                                                                                                                                 \
            /* 执行数据拷贝，注意offset已经是字节偏移 */                                                                                                         \
            *reinterpret_cast<Tmem_type *>(reinterpret_cast<char *>(dst) + dst_offset) = *reinterpret_cast<const Tmem_type *>(reinterpret_cast<const char *>(src) + src_offset); \
        }                                                                                                                                                                        \
    }

// 定义支持的约束条件数量组合
#define DEFINE_KERNELS_BY_CONSTRAINT(block_array_size, grid_array_size) \
    DEFINE_KERNELS_BY_TYPE(0, block_array_size, grid_array_size)        \
    DEFINE_KERNELS_BY_TYPE(1, block_array_size, grid_array_size)        \
    DEFINE_KERNELS_BY_TYPE(2, block_array_size, grid_array_size)

// 定义支持的类型
#define DEFINE_KERNELS_BY_TYPE(constraint_num, block_array_size, grid_array_size)      \
    DEFINE_REARRANGE_KERNEL(uchar1, constraint_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(uchar2, constraint_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(float1, constraint_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(float2, constraint_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(float4, constraint_num, block_array_size, grid_array_size) \
    DEFINE_REARRANGE_KERNEL(double4, constraint_num, block_array_size, grid_array_size)

// 与 MAX_BLOCK_ARRAY_SIZE 和 MAX_GRID_ARRAY_SIZE 耦合，需要同时修改
// 为1-5和1-5的所有组合生成内核
DEFINE_KERNELS_BY_CONSTRAINT(1, 1)
DEFINE_KERNELS_BY_CONSTRAINT(1, 2)
DEFINE_KERNELS_BY_CONSTRAINT(1, 3)
DEFINE_KERNELS_BY_CONSTRAINT(1, 4)
DEFINE_KERNELS_BY_CONSTRAINT(1, 5)
DEFINE_KERNELS_BY_CONSTRAINT(2, 1)
DEFINE_KERNELS_BY_CONSTRAINT(2, 2)
DEFINE_KERNELS_BY_CONSTRAINT(2, 3)
DEFINE_KERNELS_BY_CONSTRAINT(2, 4)
DEFINE_KERNELS_BY_CONSTRAINT(2, 5)
DEFINE_KERNELS_BY_CONSTRAINT(3, 1)
DEFINE_KERNELS_BY_CONSTRAINT(3, 2)
DEFINE_KERNELS_BY_CONSTRAINT(3, 3)
DEFINE_KERNELS_BY_CONSTRAINT(3, 4)
DEFINE_KERNELS_BY_CONSTRAINT(3, 5)
DEFINE_KERNELS_BY_CONSTRAINT(4, 1)
DEFINE_KERNELS_BY_CONSTRAINT(4, 2)
DEFINE_KERNELS_BY_CONSTRAINT(4, 3)
DEFINE_KERNELS_BY_CONSTRAINT(4, 4)
DEFINE_KERNELS_BY_CONSTRAINT(4, 5)
DEFINE_KERNELS_BY_CONSTRAINT(5, 1)
DEFINE_KERNELS_BY_CONSTRAINT(5, 2)
DEFINE_KERNELS_BY_CONSTRAINT(5, 3)
DEFINE_KERNELS_BY_CONSTRAINT(5, 4)
DEFINE_KERNELS_BY_CONSTRAINT(5, 5)

// 准备参数结构体
struct RearrangeParams {
    std::vector<ARRAY_TYPE_SIZE> block_len;
    std::vector<ARRAY_TYPE_STRIDE> src_block_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_block_stride;
    std::vector<ARRAY_TYPE_SIZE> grid_len;
    std::vector<ARRAY_TYPE_STRIDE> src_grid_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_grid_stride;
    size_t block_dim;
    size_t block_len_total;
    std::vector<Constraint<ARRAY_TYPE_SIZE>> constraints;
    size_t unit_size;
};

utils::Result<void *> getRearrangeKernel(const RearrangeParams &params) {
    auto grid_num = params.grid_len.size();
    auto block_num = params.block_len.size();
    auto constraint_num = params.constraints.size();
    auto unit_size = params.unit_size;

    CHECK_OR_RETURN(grid_num <= MAX_GRID_ARRAY_SIZE && grid_num != 0, INFINI_STATUS_BAD_PARAM);
    CHECK_OR_RETURN(block_num <= MAX_BLOCK_ARRAY_SIZE && block_num != 0, INFINI_STATUS_BAD_PARAM);

    CHECK_OR_RETURN(constraint_num <= 2, INFINI_STATUS_BAD_PARAM);

    auto block_len = params.block_len.data();
    auto src_block_stride = params.src_block_stride.data();
    auto dst_block_stride = params.dst_block_stride.data();
    auto grid_len = params.grid_len.data();
    auto src_grid_stride = params.src_grid_stride.data();
    auto dst_grid_stride = params.dst_grid_stride.data();
    auto constrain = params.constraints.data();

    void *kernel_func = nullptr;
#define GET_REARRANGE_KERNEL(Tmem_type, block_array_size, grid_array_size, constraint_num) \
    kernel_func = (void *)rearrange_unit_##Tmem_type##_block_##block_array_size##_grid_##grid_array_size##_constrain_##constraint_num;

#define GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, constraint_num)   \
    switch (unit_size) {                                                                  \
    case 1:                                                                               \
        GET_REARRANGE_KERNEL(uchar1, block_array_size, grid_array_size, constraint_num);  \
        break;                                                                            \
    case 2:                                                                               \
        GET_REARRANGE_KERNEL(uchar2, block_array_size, grid_array_size, constraint_num);  \
        break;                                                                            \
    case 4:                                                                               \
        GET_REARRANGE_KERNEL(float1, block_array_size, grid_array_size, constraint_num);  \
        break;                                                                            \
    case 8:                                                                               \
        GET_REARRANGE_KERNEL(float2, block_array_size, grid_array_size, constraint_num);  \
        break;                                                                            \
    case 16:                                                                              \
        GET_REARRANGE_KERNEL(float4, block_array_size, grid_array_size, constraint_num);  \
        break;                                                                            \
    case 32:                                                                              \
        GET_REARRANGE_KERNEL(double4, block_array_size, grid_array_size, constraint_num); \
        break;                                                                            \
    default:                                                                              \
        return INFINI_STATUS_BAD_PARAM;                                                   \
    }

#define GET_REARRANGE_KERNEL_BY_CONSTRAINT(block_array_size, grid_array_size) \
    switch (constraint_num) {                                                 \
    case 0:                                                                   \
        GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, 0);   \
        break;                                                                \
    case 1:                                                                   \
        GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, 1);   \
        break;                                                                \
    case 2:                                                                   \
        GET_REARRANGE_KERNEL_BY_TYPE(block_array_size, grid_array_size, 2);   \
        break;                                                                \
    }

#define GET_REARRANGE_KERNEL_BY_GRID_NUM(block_array_size)       \
    switch (grid_num) {                                          \
    case 1:                                                      \
        GET_REARRANGE_KERNEL_BY_CONSTRAINT(block_array_size, 1); \
        break;                                                   \
    case 2:                                                      \
        GET_REARRANGE_KERNEL_BY_CONSTRAINT(block_array_size, 2); \
        break;                                                   \
    case 3:                                                      \
        GET_REARRANGE_KERNEL_BY_CONSTRAINT(block_array_size, 3); \
        break;                                                   \
    case 4:                                                      \
        GET_REARRANGE_KERNEL_BY_CONSTRAINT(block_array_size, 4); \
        break;                                                   \
    case 5:                                                      \
        GET_REARRANGE_KERNEL_BY_CONSTRAINT(block_array_size, 5); \
        break;                                                   \
    }

#define GET_REARRANGE_KERNEL_BY_BLOCK_NUM    \
    switch (block_num) {                     \
    case 1:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(1); \
        break;                               \
    case 2:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(2); \
        break;                               \
    case 3:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(3); \
        break;                               \
    case 4:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(4); \
        break;                               \
    case 5:                                  \
        GET_REARRANGE_KERNEL_BY_GRID_NUM(5); \
        break;                               \
    }

    GET_REARRANGE_KERNEL_BY_BLOCK_NUM

    return utils::Result<void *>(kernel_func);
}
#endif // __REARRANGE_MACA_KERNEL_H__
