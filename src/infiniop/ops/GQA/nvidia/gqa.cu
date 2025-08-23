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
	  
		  // --- 共享内存设置 ---
		  // 启动此内核的主机代码必须提供正确的大小: 2 * BLOCK_SIZE * HEAD_SIZE * sizeof(T)
		  extern __shared__ char shared_mem_char_array[];
		  T* k_tile = reinterpret_cast<T*>(shared_mem_char_array);
		  T* v_tile = k_tile + BLOCK_SIZE * HEAD_SIZE;
	  
		  // --- 块与头映射 ---
		  const int block_idx = blockIdx.x;
		  const int batch_idx = block_idx / num_q_heads;
		  const int q_head_idx = block_idx % num_q_heads;
	  
		  // --- GQA 头映射 ---
		  const int gqa_group_size = num_q_heads / num_kv_heads;
		  const int kv_head_idx = q_head_idx / gqa_group_size;
	  
		  // --- 全局内存基地址指针 ---
		  const size_t q_head_offset = (size_t)(batch_idx * num_q_heads + q_head_idx) * seq_len * HEAD_SIZE;
		  const size_t k_head_offset = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_SIZE;
		  const size_t v_head_offset = (size_t)(batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_SIZE;
		  
		  const T* k_cache = key + k_head_offset;
		  const T* v_cache = value + v_head_offset;
		  
		  const float scale = rsqrtf(static_cast<float>(HEAD_SIZE));
	  
		  // =================================================================
		  // 关键架构升级: 引入外层查询循环 (Query Tiling)
		  // 这个循环使得一个线程块可以处理任意长度的序列。
		  // 块内的线程会协同处理一个大小为 BLOCK_SIZE 的查询块，然后继续处理下一个。
		  // =================================================================
		  for (int q_base = 0; q_base < seq_len; q_base += BLOCK_SIZE) {
			  // --- 动态计算当前线程负责的查询索引 t_q ---
			  const int t_q = q_base + threadIdx.x;
	  
			  // 如果当前查询索引超出了序列实际长度，则此线程在本轮及后续轮次无任务。
			  if (t_q >= seq_len) {
				  break; 
			  }
	  
			  // --- 为当前查询 t_q 初始化累加器 ---
			  // 这些变量必须在查询循环内部，因为每个线程会处理多个不同的查询。
			  float output_acc[HEAD_SIZE];
			  #pragma unroll
			  for (int i = 0; i < HEAD_SIZE; ++i) { output_acc[i] = 0.0f; }
			  float max_score = -FLT_MAX;
			  float score_sum = 0.0f;
			  
			  const T* q_vec = query + q_head_offset + (size_t)t_q * HEAD_SIZE;
	  
			  // --- 内层循环: 迭代处理 K/V 块 ---
			  // 对于当前的查询 t_q，遍历其因果历史中的所有 K/V 块。
			  for (int kv_base = 0; kv_base <= t_q; kv_base += BLOCK_SIZE) {
				  // 协同加载 K/V tile
				  for (int i = threadIdx.x; i < BLOCK_SIZE * HEAD_SIZE; i += blockDim.x) {
					  const int vec_idx_in_tile = i / HEAD_SIZE;
					  const int elem_idx_in_vec = i % HEAD_SIZE;
					  const int global_vec_idx = kv_base + vec_idx_in_tile;
	  
					  if (global_vec_idx < seq_len) {
						  const size_t cache_offset = (size_t)global_vec_idx * HEAD_SIZE + elem_idx_in_vec;
						  k_tile[i] = k_cache[cache_offset];
						  v_tile[i] = v_cache[cache_offset];
					  } else {
						  k_tile[i] = static_cast<T>(0.0f);
						  v_tile[i] = static_cast<T>(0.0f);
					  }
				  }
				  __syncthreads(); // 等待 tile 加载完成
	  
				  // 遍历 tile 内的 K/V 向量进行计算
				  for (int k_idx_in_tile = 0; k_idx_in_tile < BLOCK_SIZE; ++k_idx_in_tile) {
					  const int global_k_idx = kv_base + k_idx_in_tile;
	  
					  // 应用因果掩码
					  if (global_k_idx <= t_q) {
						  const T* k_vec = &k_tile[k_idx_in_tile * HEAD_SIZE];
						  
						  float score = 0.0f;
						  #pragma unroll
						  for (int i = 0; i < HEAD_SIZE; ++i) {
							  score += static_cast<float>(q_vec[i]) * static_cast<float>(k_vec[i]);
						  }
						  score *= scale;
	  
						  if (score > max_score) {
							  float old_max_score = max_score;
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
	  
			  // --- 最终归一化并写回当前 t_q 的结果 ---
			  const float inv_sum = (score_sum > 1e-6f) ? 1.0f / score_sum : 0.0f;
			  T* output_vec = output + q_head_offset + (size_t)t_q * HEAD_SIZE;
			  #pragma unroll
			  for (int i = 0; i < HEAD_SIZE; ++i) {
				  output_vec[i] = static_cast<T>(output_acc[i] * inv_sum);
			  }
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
 