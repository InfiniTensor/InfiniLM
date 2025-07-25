#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "topk.cuh"
#include <cub/util_type.cuh>
#include <cub/cub.cuh>
#include <cfloat>

namespace op::topk::nvidia {

struct Descriptor::Opaque {
	std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
	delete _opaque;
}

infiniStatus_t Descriptor::create(
	infiniopHandle_t handle, Descriptor **desc_ptr,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t output_val_desc,
	infiniopTensorDescriptor_t output_ind_desc,
	infiniopTensorDescriptor_t bias_desc, int k, TopKStrategy strategy,
	int n_group, int topk_group) {
	auto result =
		TopKInfo::create(input_desc, output_val_desc, output_ind_desc,
							bias_desc, k, strategy, n_group, topk_group);
	CHECK_RESULT(result);

	*desc_ptr = new Descriptor(result. take(), nullptr, INFINI_DEVICE_NVIDIA,
								handle->device_id);
	return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::getWorkspaceSize() const { return _info.workspace_size; }

template <typename T>
__global__ void add_bias_kernel(T *data, const T *bias, int num_tokens,
                                int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * num_experts) {
        int expert_idx = idx % num_experts;
        data[idx] += bias[expert_idx];
    }
}

template <typename T>
__global__ void
cast_add_bias_sigmoid_kernel(float *output, const T *input, const T *bias,
                             int num_tokens, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = num_tokens * num_experts;
    if (idx < total_size) {
        float val = static_cast<float>(input[idx]);
        if (bias) {
            int expert_idx = idx % num_experts;
            val += static_cast<float>(bias[expert_idx]);
        }
        output[idx] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void get_topk_indices_grouped_kernel(
    const float *scores, int *topk_indices, int num_tokens, int num_experts,
    int n_group, int topk_group, int k, float *group_scores_workspace,
    int *group_idx_workspace, char *score_mask_workspace) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    // 1. Compute group scores
    for (int i = threadIdx.x; i < n_group; i += blockDim.x) {
        float max_score1 = -1.0f, max_score2 = -1.0f;
        int experts_per_group = num_experts / n_group;
        for (int j = 0; j < experts_per_group; ++j) {
            float score =
                scores[token_idx * num_experts + i * experts_per_group + j];
            if (score > max_score1) {
                max_score2 = max_score1;
                max_score1 = score;
            } else if (score > max_score2) {
                max_score2 = score;
            }
        }
        group_scores_workspace[token_idx * n_group + i] = max_score1 + max_score2;
    }
    __syncthreads();

    // 2. Top-k groups
    if (threadIdx.x == 0) {
        for (int i = 0; i < topk_group; ++i) {
            float max_val = -1.0f;
            int max_idx = -1;
            for (int j = 0; j < n_group; ++j) {
                bool already_selected = false;
                for (int l = 0; l < i; ++l) {
                    if (group_idx_workspace[token_idx * topk_group + l] == j) {
                        already_selected = true;
                        break;
                    }
                }
                if (!already_selected &&
                    group_scores_workspace[token_idx * n_group + j] > max_val) {
                    max_val = group_scores_workspace[token_idx * n_group + j];
                    max_idx = j;
                }
            }
            group_idx_workspace[token_idx * topk_group + i] = max_idx;
        }

        // 3. Create score mask
        for (int i = 0; i < num_experts; ++i) {
            score_mask_workspace[token_idx * num_experts + i] = 0;
        }
        for (int i = 0; i < topk_group; ++i) {
            int group_idx = group_idx_workspace[token_idx * topk_group + i];
            int experts_per_group = num_experts / n_group;
            for (int j = 0; j < experts_per_group; ++j) {
                score_mask_workspace[token_idx * num_experts +
                                     group_idx * experts_per_group + j] = 1;
            }
        }

        // 4. Final top-k on masked scores
        for (int i = 0; i < k; ++i) {
            float max_val = -1.0f;
            int max_idx = -1;
            for (int j = 0; j < num_experts; ++j) {
                if (score_mask_workspace[token_idx * num_experts + j]) {
                    bool already_selected = false;
                    for (int l = 0; l < i; ++l) {
                        if (topk_indices[token_idx * k + l] == j) {
                            already_selected = true;
                            break;
                        }
                    }
                    if (!already_selected &&
                        scores[token_idx * num_experts + j] > max_val) {
                        max_val = scores[token_idx * num_experts + j];
                        max_idx = j;
                    }
                }
            }
            topk_indices[token_idx * k + i] = max_idx;
        }
    }
}

template <typename T, typename IndType>
void get_topk_indices_kernel(const T *scores, const T *bias,
                             IndType *topk_indices, int num_tokens,
                             int num_experts, int n_group, int topk_group, int k,
                             void *workspace, cudaStream_t stream) {

    float *scores_for_choice = static_cast<float *>(workspace);
    float *group_scores = scores_for_choice + num_tokens * num_experts;
    int *group_idx =
        reinterpret_cast<int *>(group_scores + num_tokens * n_group);
    char *score_mask =
        reinterpret_cast<char *>(group_idx + num_tokens * topk_group);

    dim3 grid_dim_fuse((num_tokens * num_experts + 255) / 256);
    dim3 block_dim_fuse(256);
    cast_add_bias_sigmoid_kernel<T><<<grid_dim_fuse, block_dim_fuse, 0, stream>>>(
        scores_for_choice, scores, bias, num_tokens, num_experts);

    dim3 block_dim_grouped(256);
    dim3 grid_dim_grouped(num_tokens);
    get_topk_indices_grouped_kernel<<<grid_dim_grouped, block_dim_grouped, 0,
                                     stream>>>(
        scores_for_choice, (int *)topk_indices, num_tokens, num_experts,
        n_group, topk_group, k, group_scores, group_idx, score_mask);
}

template <typename T, typename IndType>
__global__ void
gather_topk_weights_kernel(const float *scores_for_choice,
                           const IndType *topk_indices, T *topk_weights,
                           int num_tokens, int num_experts, int k) {
    int token_idx = blockIdx.x;
    int k_idx = threadIdx.x;

    if (token_idx < num_tokens && k_idx < k) {
        int expert_index = topk_indices[token_idx * k + k_idx];
        topk_weights[token_idx * k + k_idx] = static_cast<T>(
            scores_for_choice[token_idx * num_experts + expert_index]);
    }
}

template <typename T>
__global__ void normalize_topk_weights_kernel(T *topk_weights, int num_tokens,
                                              int k) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += static_cast<float>(topk_weights[token_idx * k + i]);
    }

    if (sum > 1e-20) {
        for (int i = 0; i < k; ++i) {
            float val = static_cast<float>(topk_weights[token_idx * k + i]);
            topk_weights[token_idx * k + i] = static_cast<T>(val / sum);
        }
    }
}



template <typename T, typename IndType>
void deepseek_v3_topk_router(const void *input, void *output_val,
                             void *output_ind, const void *bias,
                             void *workspace, const TopKInfo &info,
                             cudaStream_t stream) {

    const int num_tokens = info.num_tokens;
    const int num_experts = info.num_experts;
    const int k = info.k;

    get_topk_indices_kernel<T, IndType>(
        static_cast<const T *>(input), static_cast<const T *>(bias),
        static_cast<IndType *>(output_ind), info.num_tokens,
        info.num_experts, info.n_group, info.topk_group, info.k,
        workspace, stream);

    float *scores_for_choice = static_cast<float *>(workspace);
    dim3 grid_dim_gather(num_tokens);
    dim3 block_dim_gather(k);
    gather_topk_weights_kernel<T, IndType><<<grid_dim_gather, block_dim_gather,
                                             0, stream>>>(
        scores_for_choice, static_cast<const IndType *>(output_ind),
        static_cast<T *>(output_val), num_tokens, num_experts, k);

    dim3 grid_dim_norm(num_tokens);
    normalize_topk_weights_kernel<T>
        <<<grid_dim_norm, 1, 0, stream>>>(static_cast<T *>(output_val),
                                          num_tokens, k);
}

template <typename T>
__global__ void softmax_kernel(T *data, int num_tokens, int num_experts) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    float max_val = -FLT_MAX;
    for (int i = 0; i < num_experts; ++i) {
        max_val =
            max(max_val, static_cast<float>(data[token_idx * num_experts + i]));
    }

    float sum = 0.0f;
    for (int i = 0; i < num_experts; ++i) {
        sum +=
            expf(static_cast<float>(data[token_idx * num_experts + i]) - max_val);
    }

    if (sum > 1e-20) {
        for (int i = 0; i < num_experts; ++i) {
            float val = static_cast<float>(data[token_idx * num_experts + i]);
            data[token_idx * num_experts + i] =
                static_cast<T>(expf(val - max_val) / sum);
        }
    }
}

template <typename T, typename IndType>
__global__ void standard_topk_kernel(const T *scores, IndType *topk_indices,
                                     T *topk_weights, int num_tokens,
                                     int num_experts, int k) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    for (int i = 0; i < k; ++i) {
        float max_val = -1.0f;
        int max_idx = -1;
        for (int j = 0; j < num_experts; ++j) {
            bool already_selected = false;
            for (int l = 0; l < i; ++l) {
                if (topk_indices[token_idx * k + l] == j) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected &&
                static_cast<float>(scores[token_idx * num_experts + j]) >
                    max_val) {
                max_val = static_cast<float>(scores[token_idx * num_experts + j]);
                max_idx = j;
            }
        }
        topk_indices[token_idx * k + i] = max_idx;
        topk_weights[token_idx * k + i] = static_cast<T>(max_val);
    }
}

template <typename T, typename IndType>
void standard_topk_router(const void *input, void *output_val,
                          void *output_ind, const void *bias, void *workspace,
                          const TopKInfo &info, cudaStream_t stream) {
    const int num_tokens = info.num_tokens;
    const int num_experts = info.num_experts;
    const int k = info.k;

    cudaMemcpyAsync(workspace, input, num_tokens * num_experts * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    if (bias) {
        dim3 grid_dim_add((num_tokens * num_experts + 255) / 256);
        dim3 block_dim_add(256);
        add_bias_kernel<T><<<grid_dim_add, block_dim_add, 0, stream>>>(
            static_cast<T *>(workspace), static_cast<const T *>(bias),
            num_tokens, num_experts);
    }

    dim3 grid_dim_softmax(num_tokens);
    softmax_kernel<T>
        <<<grid_dim_softmax, 256, 0, stream>>>(static_cast<T *>(workspace),
                                               num_tokens, num_experts);

    dim3 grid_dim_topk(num_tokens);
    standard_topk_kernel<T, IndType><<<grid_dim_topk, k, 0, stream>>>(
        static_cast<const T *>(workspace), static_cast<IndType *>(output_ind),
        static_cast<T *>(output_val), num_tokens, num_experts, k);
}

infiniStatus_t Descriptor::calculate(const void *input, void *output_val,
                                     void *output_ind, const void *bias,
                                     void *workspace, void *stream) const {
    if (_info.workspace_size > 0 && workspace == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    if (_info.strategy == DEEPSEEK_V3) {
        if (_info.data_type == INFINI_DTYPE_F32) {
            deepseek_v3_topk_router<float, int>(input, output_val, output_ind,
                                                bias, workspace, _info,
                                                (cudaStream_t)stream);
        } else if (_info.data_type == INFINI_DTYPE_F16) {
            deepseek_v3_topk_router<half, int>(
                input, output_val, output_ind, bias, workspace, _info,
                (cudaStream_t)stream);
        } else if (_info.data_type == INFINI_DTYPE_BF16) {
            deepseek_v3_topk_router<__nv_bfloat16, int>(
                input, output_val, output_ind, bias, workspace, _info,
                (cudaStream_t)stream);
        } else {
			printf("Unsupported data type for TopK\n");
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    } else { // STANDARD_SOFTMAX
        if (_info.data_type == INFINI_DTYPE_F32) {
            standard_topk_router<float, int>(input, output_val, output_ind,
                                             bias, workspace, _info,
                                             (cudaStream_t)stream);
        } else if (_info.data_type == INFINI_DTYPE_F16) {
            standard_topk_router<half, int>(input, output_val, output_ind,
                                            bias, workspace, _info,
                                            (cudaStream_t)stream);
        } else if (_info.data_type == INFINI_DTYPE_BF16) {
            standard_topk_router<__nv_bfloat16, int>(
                input, output_val, output_ind, bias, workspace, _info,
                (cudaStream_t)stream);
        } else {
            printf("Unsupported data type for TopK\n");
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::topk::nvidia 