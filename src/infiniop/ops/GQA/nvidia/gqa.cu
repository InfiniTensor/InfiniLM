/**
 * @file gqa.cu
 * @brief CUDA implementation of the Grouped-Query Attention (GQA) operator.
 *
 * This file contains the complete implementation for a GQA operator, designed
 * to integrate into a framework similar to the one suggested by the user's
 * reference code. It includes descriptor management, data type and hardware
 * capability checks, and a CUDA kernel for performing the attention calculation.
 */
 #include "../../../devices/nvidia/nvidia_handle.cuh"
 #include "gqa.cuh"
 #include <cuda_fp16.h>    // For __half (FP16)
 #include <cuda_bf16.h>   // For __nv_bfloat16 (BF16)
 #include <iostream>      // For error messages
 #include <memory>        // For std::shared_ptr
 #include <cmath>         // For sqrtf
 #include <float.h>       // For FLT_MAX
 // --- GQA Operator Implementation ---
 namespace op::gqa::nvidia {

	struct Descriptor::Opaque {
		std::shared_ptr<device::nvidia::Handle::Internal> internal;
	};
	
	Descriptor::~Descriptor() {
		delete _opaque;
	}
 // Maximum dimensions supported by this simple kernel implementation.
 // The block size in the kernel launch is set to `seq_len`, which is limited to 1024 threads.
 // The kernel uses static arrays on the stack, so we must limit their size.
 template <typename T>
 infiniStatus_t dispatch_by_head_size(const GQAInfo &_info, const void *q, const void *k, const void *v,
	void *output, cudaStream_t stream);

 infiniStatus_t Descriptor::create(
	infiniopHandle_t handle_, Descriptor **desc_ptr,
	infiniopTensorDescriptor_t q_desc,
	infiniopTensorDescriptor_t k_desc,
	infiniopTensorDescriptor_t v_desc,
	infiniopTensorDescriptor_t output_desc) {
	auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
	auto dtype = q_desc->dtype();
	if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32 && dtype != INFINI_DTYPE_BF16) {
		return INFINI_STATUS_BAD_TENSOR_DTYPE;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, handle->device);

	if (dtype == INFINI_DTYPE_F16 && prop.major < 7) {
		std::cerr << "FP16 is not supported on devices with compute capability < 7.0." << std::endl;
		return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
	}
	if (dtype == INFINI_DTYPE_BF16 && prop.major < 8) {
		std::cerr << "BF16 is not supported on devices with compute capability < 8.0." << std::endl;
		return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
	}

	auto result = GQAInfo::create(q_desc, k_desc, v_desc, output_desc);
	if (!result) {
		return INFINI_STATUS_BAD_TENSOR_SHAPE;
	}

	// Check if dimensions exceed the limits of this simple kernel

	*desc_ptr = new Descriptor(dtype, result.take(), 0,
								new Opaque{handle->internal()}, handle->device,
								handle_->device_id);
	return INFINI_STATUS_SUCCESS;
}
 /**
  * @brief CUDA kernel for Grouped-Query Attention.
  *
  * This kernel calculates the attention output for a single query token.
  * Each thread in a block is responsible for one query token (t_q).
  * The grid is structured so that each block handles a specific (batch, query_head) pair.
  *
  * The calculation follows these steps for each thread:
  * 1. Computes dot-product scores between its query vector and all key vectors.
  * 2. Applies scaling and performs softmax on the scores to get attention weights.
  * 3. Computes the weighted sum of value vectors using the attention weights.
  * 4. Writes the final output vector to global memory.
  *
  * @note This is a straightforward implementation and is not optimized for performance
  * in the same way as tiled approaches like FlashAttention. It uses stack-allocated
  * arrays, hence the `GQA_MAX_SEQ_LEN` and `GQA_MAX_HEAD_SIZE` limitations.
  */
  template <typename T, int BLOCK_SIZE, int HEAD_SIZE>
  __global__ void kernel_gqa_one_pass(
	  const T* query, const T* key, const T* value, T* output,
	  const int batch_size, const int seq_len,
	  const int num_q_heads, const int num_kv_heads) {
  
	  // 为 K 和 V 矩阵的瓦片(tile)准备共享内存
	  extern __shared__ char shared_mem_char_array[];
	  T* k_tile = reinterpret_cast<T*>(shared_mem_char_array);
	  T* v_tile = k_tile + BLOCK_SIZE * HEAD_SIZE;
  
	  // 每个 block 处理一个 (batch, query_head) 对
	  const int block_idx = blockIdx.x;
	  const int batch_idx = block_idx / num_q_heads;
	  const int q_head_idx = block_idx % num_q_heads;
  
	  // 每个线程负责计算一个 query token (一行)的输出
	  const int t_q = threadIdx.x;
	  if (t_q >= seq_len) return;
  
	  // 根据 GQA 确定对应的 key/value head 索引
	  const int gqa_group_size = num_q_heads / num_kv_heads;
	  const int kv_head_idx = q_head_idx / gqa_group_size;
  
	  // 定位到当前 head 的 Q, K, V 数据指针
	  const size_t q_offset = (size_t)(batch_idx * num_q_heads + q_head_idx) * seq_len * HEAD_SIZE;
	  const size_t k_offset = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_SIZE;
	  const size_t v_offset = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_SIZE;
  
	  const T* q_vec = query + q_offset + (size_t)t_q * HEAD_SIZE;
	  const T* k_cache = key + k_offset;
	  const T* v_cache = value + v_offset;
  
	  // --- 优化 1: 使用寄存器/栈(由编译器决定)为每个线程分配累加器 ---
	  // HEAD_SIZE 是模板参数，编译器可以高效地优化这个数组
	  float output_acc[HEAD_SIZE];
	  #pragma unroll
	  for (int i = 0; i < HEAD_SIZE; ++i) {
		  output_acc[i] = 0.0f;
	  }
  
	  // --- 优化 2: 实现单遍(One-Pass)在线 Softmax ---
	  float max_score = -FLT_MAX;
	  float score_sum = 0.0f;
	  const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_SIZE));
  
	  // 在单个循环中处理所有 KV 块
	  for (int kv_base = 0; kv_base < seq_len; kv_base += BLOCK_SIZE) {
		  // 协作加载 K 和 V 的瓦片到共享内存
		  __syncthreads();
		  for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_SIZE; i += blockDim.x) {
			  const int vec_idx_in_tile = i / HEAD_SIZE;
			  const int elem_idx_in_vec = i % HEAD_SIZE;
			  const int global_vec_idx = kv_base + vec_idx_in_tile;
  
			  if (global_vec_idx < seq_len) {
				  const size_t cache_offset = (size_t)global_vec_idx * HEAD_SIZE + elem_idx_in_vec;
				  k_tile[i] = k_cache[cache_offset];
				  v_tile[i] = v_cache[cache_offset];
			  } else {
				  // 填充以防止使用陈旧数据
				  k_tile[i] = static_cast<T>(0.0f);
				  v_tile[i] = static_cast<T>(0.0f);
			  }
		  }
		  __syncthreads();
  
		  // 遍历瓦片中的每个 key/value 向量
		  for (int k_idx_in_tile = 0; k_idx_in_tile < BLOCK_SIZE; ++k_idx_in_tile) {
			  if (kv_base + k_idx_in_tile < seq_len) {
				  const T* k_vec = &k_tile[k_idx_in_tile * HEAD_SIZE];
				  
				  // 计算 QK 点积
				  float score = 0.0f;
				  #pragma unroll
				  for (int i = 0; i < HEAD_SIZE; ++i) {
					  score += static_cast<float>(q_vec[i]) * static_cast<float>(k_vec[i]);
				  }
				  score *= scale;
  
				  // --- 在线 Softmax 的核心逻辑 ---
				  if (score > max_score) {
					  const float old_max_score = max_score;
					  max_score = score;
					  
					  // 当找到新的最大值时，对历史累加值进行缩放
					  const float rescale_factor = expf(old_max_score - max_score);
					  score_sum *= rescale_factor;
					  #pragma unroll
					  for (int i = 0; i < HEAD_SIZE; ++i) {
						  output_acc[i] *= rescale_factor;
					  }
				  }
  
				  // 计算当前 key 的权重并累加
				  const float attention_weight = expf(score - max_score);
				  score_sum += attention_weight;
  
				  const T* v_vec = &v_tile[k_idx_in_tile * HEAD_SIZE];
				  #pragma unroll
				  for (int i = 0; i < HEAD_SIZE; ++i) {
					  output_acc[i] += attention_weight * static_cast<float>(v_vec[i]);
				  }
			  }
		  }
	  }
  
	  // 最终归一化并写回全局内存
	  const float inv_sum = 1.0f / score_sum;
	  T* output_vec = output + q_offset + (size_t)t_q * HEAD_SIZE;
	  #pragma unroll
	  for (int i = 0; i < HEAD_SIZE; ++i) {
		  output_vec[i] = static_cast<T>(output_acc[i] * inv_sum);
	  }
  }
  // --- 新增: 分发器函数 ---
// This function reads the runtime head_size and dispatches to the
// correctly templated version of the launch_kernel function.

  // Helper function to launch the templated kernel
  // Helper function to launch the templated kernel
// --- 修改点 1: 增加 HEAD_SIZE 模板参数 ---
template <typename T, int HEAD_SIZE>
void launch_kernel(const GQAInfo &_info,
                   const void *q, const void *k, const void *v, void *output,
                   cudaStream_t stream) {
    // Each block processes one (batch, q_head) pair.
    dim3 grid(_info.batch_size * _info.num_q_heads);
    
    // Block size can be smaller than seq_len. The kernel handles this tiling internally.
    // 256 is a reasonable default. A better approach might be to tune this.
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);

    // Calculate required shared memory: (K_tile + V_tile) * sizeof(T)
    size_t shared_mem_bytes = 2 * BLOCK_SIZE * HEAD_SIZE * sizeof(T);

    // --- 修改点 2: 将 HEAD_SIZE 作为模板参数传入，并移除第9个运行时参数 ---
    kernel_gqa_one_pass<T, BLOCK_SIZE, HEAD_SIZE><<<grid, block, shared_mem_bytes, stream>>>(
        static_cast<const T *>(q),
        static_cast<const T *>(k),
        static_cast<const T *>(v),
        static_cast<T *>(output),
        _info.batch_size, _info.seq_len, _info.num_q_heads, _info.num_kv_heads
    );
}
infiniStatus_t
Descriptor::calculate(const void *q, const void *k, const void *v,
                      void *output, void *stream) const {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // --- 修改点 3: 调用分发器，而不是直接调用 launch_kernel ---
    switch (_info.data_type) {
		case INFINI_DTYPE_F32:
			// --- FIX: Add the full namespace to the function call ---
			return op::gqa::nvidia::dispatch_by_head_size<float>(_info, q, k, v, output, cuda_stream);
		case INFINI_DTYPE_F16:
			// --- FIX: Add the full namespace to the function call ---
			return op::gqa::nvidia::dispatch_by_head_size<half>(_info, q, k, v, output, cuda_stream);
		case INFINI_DTYPE_BF16:
			// --- FIX: Add the full namespace to the function call ---
			return op::gqa::nvidia::dispatch_by_head_size<__nv_bfloat16>(_info, q, k, v, output, cuda_stream);
		default:
			return INFINI_STATUS_BAD_TENSOR_DTYPE;
		}
}

template <typename T>
infiniStatus_t dispatch_by_head_size(const GQAInfo &_info, const void *q, const void *k, const void *v,
                                     void *output, cudaStream_t stream) {
    switch (_info.head_size) {
    case 64:
        launch_kernel<T, 64>(_info, q, k, v, output, stream);
        break;
    case 128:
        launch_kernel<T, 128>(_info, q, k, v, output, stream);
        break;
    case 256:
        launch_kernel<T, 256>(_info, q, k, v, output, stream);
        break;
    default:
        // This case should ideally be caught by the check in `Descriptor::create`
        return INFINI_STATUS_BAD_PARAM;
    }
    return INFINI_STATUS_SUCCESS;
}
 
 } // namespace op::gqa::nvidia
 