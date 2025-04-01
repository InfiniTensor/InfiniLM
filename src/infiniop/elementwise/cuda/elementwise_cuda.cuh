// #ifndef __INFINIOP_ELEMENTWISE_CUDA_H__
// #define __INFINIOP_ELEMENTWISE_CUDA_H__

// #include "../../devices/cuda/cuda_common.cuh"
// #include "../elementwise.h"

// #define ELEMENTWISE_CUDA_OPAQUE(OP)                               \
//                                                                   \
//     namespace op::OP::cuda {                                      \
//     struct Descriptor::Opaque {                                   \
//         std::shared_ptr<device::cuda::Handle::Internal> internal; \
//     };                                                            \
//                                                                   \
//     Descriptor::~Descriptor() {                                   \
//         delete _opaque;                                           \
//     }                                                             \
//     } // namespace op::elementwise::cuda

// namespace op::common_cuda::elementwise_op {

// // Perform elementwise operation when all inputs have the same type
// template <size_t BLOCK_SIZE, typename Op, typename Tdata, size_t... Is, typename... Args>
// void _calculate_impl(const op::elementwise::ElementwiseInfo &info,
//                      void *output,
//                      const std::vector<const void *> &inputs,
//                      std::index_sequence<Is...>,
//                      Args &&...args) {

//     Tdata *out = reinterpret_cast<Tdata *>(output);
//     std::array<const Tdata *, sizeof...(Is)> ins = {reinterpret_cast<const Tdata *>(inputs[Is])...};
//     const ptrdiff_t output_size = info.output_size;

// #pragma omp parallel for
//     for (ptrdiff_t i = 0; i < output_size; ++i) {
//         size_t out_idx = info.output_contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.output_shape.data(), info.output_strides.data());

//         auto get_input_idx = [&](size_t input_id) {
//             return info.input_contiguous[input_id] ? i
//                                                    : (info.input_broadcasted[input_id]
//                                                           ? op::common_cpu::indexToReducedOffset(i, info.ndim, info.output_strides.data(), info.input_strides[input_id].data())
//                                                           : op::common_cpu::indexToOffset(i, info.ndim, info.input_shapes[input_id].data(), info.input_strides[input_id].data()));
//         };

//         if constexpr (std::is_same_v<Tdata, fp16_t>) {
//             out[out_idx] = utils::cast<fp16_t>(Op{}(utils::cast<float>(ins[Is][get_input_idx(Is)])..., std::forward<Args>(args)...));
//         } else {
//             out[out_idx] = Op{}(ins[Is][get_input_idx(Is)]..., std::forward<Args>(args)...);
//         }
//     }
// }

// template <size_t BLOCK_SIZE, typename Op, typename Tdata, size_t... Is, typename... Args>
// void calculate_impl(const op::elementwise::ElementwiseInfo &info,
//                     void *output,
//                     const std::vector<const void *> &inputs,
//                     std::index_sequence<Is...>,
//                     Args &&...args) {

//     if (info.output_size == 0) {
//         return;
//     }
//     Tdata *out = reinterpret_cast<Tdata *>(output);
//     std::array<const Tdata *, sizeof...(Is)> inputs_vec = {reinterpret_cast<const Tdata *>(inputs[Is])...};

//     dim3 blockDims = dim3(std::min(static_cast<uint64_t>(BLOCK_SIZE), info.output_size));
//     dim3 gridDims = dim3(std::min(ROUND_UP_DIV(info.output_size, blockDims.x), desc->max_grid_size));
//     uint64_t step = gridDims.x * blockDims.x;

//     _calculate_impl<BLOCK_SIZE, Op, Tdata, TIdata>(info, out, inputs_vec, Is, std::forward<Args>(args)...);
// }

// // Invoke elementwise operation when all inputs have the same type
// template <size_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
// void calculate(const op::elementwise::ElementwiseInfo &info, void *output, const std::vector<const void *> &inputs, Args &&...args) {
//     constexpr size_t N = Op::num_inputs;
//     calculate_impl<BLOCK_SIZE, Op, Tdata>(info, output, inputs, std::make_index_sequence<N>{}, std::forward<Args>(args)...);
// }

// } // namespace op::common_cuda::elementwise_op

// #endif // __INFINIOP_ELEMENTWISE_CUDA_H__