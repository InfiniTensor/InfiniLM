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
 #include <iostream>       // For error messages
 #include <memory>         // For std::shared_ptr
 #include <cmath>          // For sqrtf
 #include <cfloat>         // For FLT_MAX
 
 // --- GQA Operator Implementation ---
 namespace op::gqa::nvidia {
 
	 struct Descriptor::Opaque {
		 std::shared_ptr<device::nvidia::Handle::Internal> internal;
	 };
 
	 Descriptor::~Descriptor() {
		 delete _opaque;
	 }
 
	 // Forward declaration for the dispatcher function
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
		 
		 // Check if dimensions are supported by the templated kernel
		 auto gqa_info = result.take();
        
        // Check if dimensions are supported by the templated kernel
         const int head_size = gqa_info.head_size;
		 if (head_size != 32 && head_size != 64 && head_size != 128 && head_size != 256) {
			 std::cerr << "Unsupported head_size: " << head_size 
					   << ". This kernel only supports 32, 64, 128, or 256." << std::endl;
			 return INFINI_STATUS_BAD_PARAM;
		 }
 
 
		 *desc_ptr = new Descriptor(dtype, result.take(), 0,
									new Opaque{handle->internal()}, handle->device,
									handle_->device_id);
		 return INFINI_STATUS_SUCCESS;
	 }
 
	 /**
	  * @brief CUDA kernel for Grouped-Query Attention.
	  * ... (kernel documentation) ...
	  */
	  template <typename T, int BLOCK_SIZE, int HEAD_SIZE>
__global__ void kernel_gqa_one_pass(
    const T* query, const T* key, const T* value, T* output,
    const int batch_size, const int seq_len,
    const int num_q_heads, const int num_kv_heads) {

		extern __shared__ char shared_mem_char_array[];
		T* k_tile = reinterpret_cast<T*>(shared_mem_char_array);
		T* v_tile = k_tile + BLOCK_SIZE * HEAD_SIZE;
	
		const int block_idx = blockIdx.x;
		const int batch_idx = block_idx / num_q_heads;
		const int q_head_idx = block_idx % num_q_heads;
	
		const int t_q = threadIdx.x;
		if (t_q >= seq_len) return;
	
		const int gqa_group_size = num_q_heads / num_kv_heads;
		const int kv_head_idx = q_head_idx / gqa_group_size;
	
		const size_t q_offset = (size_t)(batch_idx * num_q_heads + q_head_idx) * seq_len * HEAD_SIZE;
		const size_t k_offset = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_SIZE;
		const size_t v_offset = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_SIZE;
	
		const T* q_vec = query + q_offset + (size_t)t_q * HEAD_SIZE;
		const T* k_cache = key + k_offset;
		const T* v_cache = value + v_offset;
	
		float output_acc[HEAD_SIZE];
		#pragma unroll
		for (int i = 0; i < HEAD_SIZE; ++i) {
			output_acc[i] = 0.0f;
		}
	
		float max_score = -FLT_MAX;
		float score_sum = 0.0f;
		const float scale = rsqrtf(static_cast<float>(HEAD_SIZE));
	
		// =================================================================
		// 关键修复: 统一循环边界以保证线程协作的正确性
		// 之前的问题是：每个线程根据自己的 t_q 决定循环次数 (for kv_base <= t_q)。
		// 这导致低 t_q 的线程提前退出循环，不再参与后续 K/V 块的共享内存加载，
		// 从而破坏了高 t_q 线程的数据一致性，导致 __syncthreads() 行为未定义。
		//
		// 现在的修复是：块内的所有线程都必须执行相同的循环次数，以确保它们
		// 始终一起协作加载数据。循环的上界应该是这个块需要处理的最大 token 索引。
		// =================================================================
		const int max_seq_len_for_block = min(seq_len, (int)blockDim.x);
		for (int kv_base = 0; kv_base < max_seq_len_for_block; kv_base += BLOCK_SIZE) {
			__syncthreads(); // 在加载新 tile 前同步
			for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_SIZE; i += blockDim.x) {
				const int vec_idx_in_tile = i / HEAD_SIZE;
				const int elem_idx_in_vec = i % HEAD_SIZE;
				const int global_vec_idx = kv_base + vec_idx_in_tile;
	
				// 只有在全局索引有效时才从全局内存加载
				if (global_vec_idx < seq_len) {
					const size_t cache_offset = (size_t)global_vec_idx * HEAD_SIZE + elem_idx_in_vec;
					k_tile[i] = k_cache[cache_offset];
					v_tile[i] = v_cache[cache_offset];
				} else {
					// 否则用 0 填充，防止使用陈旧数据
					k_tile[i] = static_cast<T>(0.0f);
					v_tile[i] = static_cast<T>(0.0f);
				}
			}
			__syncthreads(); // 确保所有数据都已加载到共享内存
	
			// 现在所有线程都拥有一个有效的 K/V tile，每个线程根据自己的 t_q 进行计算
			for (int k_idx_in_tile = 0; k_idx_in_tile < BLOCK_SIZE; ++k_idx_in_tile) {
				const int global_k_idx = kv_base + k_idx_in_tile;
	
				// 每个线程依然只处理自己能看到的数据（因果掩码）
				if (global_k_idx <= t_q) {
					const T* k_vec = &k_tile[k_idx_in_tile * HEAD_SIZE];
					
					float score = 0.0f;
					#pragma unroll
					for (int i = 0; i < HEAD_SIZE; ++i) {
						score += static_cast<float>(q_vec[i]) * static_cast<float>(k_vec[i]);
					}
					score *= scale;
	
					if (score > max_score) {
						const float old_max_score = max_score;
						max_score = score;
						
						if (old_max_score != -FLT_MAX) {
							const float rescale_factor = expf(old_max_score - max_score);
							score_sum *= rescale_factor;
							#pragma unroll
							for (int i = 0; i < HEAD_SIZE; ++i) {
								output_acc[i] *= rescale_factor;
							}
						}
					}
	
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
	
		const float inv_sum = (score_sum > 1e-6f) ? 1.0f / score_sum : 0.0f;
		T* output_vec = output + q_offset + (size_t)t_q * HEAD_SIZE;
		#pragma unroll
		for (int i = 0; i < HEAD_SIZE; ++i) {
			output_vec[i] = static_cast<T>(output_acc[i] * inv_sum);
		}
}
 
	 // Helper function to launch the templated kernel
	 template <typename T, int HEAD_SIZE>
	 void launch_kernel(const GQAInfo &_info,
						const void *q, const void *k, const void *v, void *output,
						cudaStream_t stream) {
		 // Each block processes one (batch, q_head) pair.
		 dim3 grid(_info.batch_size * _info.num_q_heads);
		 
		 // Block size can be smaller than seq_len. The kernel handles this tiling internally.
		 // 256 is a reasonable default that works well on many GPUs.
		 const int BLOCK_SIZE = 256;
		 dim3 block(BLOCK_SIZE);
 
		 // Calculate required shared memory: (K_tile + V_tile) * sizeof(T)
		 size_t shared_mem_bytes = 2 * BLOCK_SIZE * HEAD_SIZE * sizeof(T);
 
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
 
		 // Dispatch to the dispatcher function based on data type
		 switch (_info.data_type) {
			 case INFINI_DTYPE_F32:
				 return op::gqa::nvidia::dispatch_by_head_size<float>(_info, q, k, v, output, cuda_stream);
			 case INFINI_DTYPE_F16:
				 return op::gqa::nvidia::dispatch_by_head_size<half>(_info, q, k, v, output, cuda_stream);
			 case INFINI_DTYPE_BF16:
				 return op::gqa::nvidia::dispatch_by_head_size<__nv_bfloat16>(_info, q, k, v, output, cuda_stream);
			 default:
				 return INFINI_STATUS_BAD_TENSOR_DTYPE;
			 }
	 }
 
	 // Dispatcher function to handle runtime head_size
	 template <typename T>
	 infiniStatus_t dispatch_by_head_size(const GQAInfo &_info, const void *q, const void *k, const void *v,
										  void *output, cudaStream_t stream) {
		 switch (_info.head_size) {
		 case 32:
			 launch_kernel<T, 32>(_info, q, k, v, output, stream);
			 break;
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
			 return INFINI_STATUS_BAD_PARAM;
		 }
		 return INFINI_STATUS_SUCCESS;
	 }
 
 } // namespace op::gqa::nvidia
 