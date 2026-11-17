#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InfiniLMCausalConv1dLaunchArgs {
	int dtype; /* 0=f32, 1=f16, 2=bf16 */
	int request_count;
	int C;
	int T;
	int total_tokens;
	int pool_size;
	int has_bias;
	int has_cu_seqlens;
	int cu_seqlens_i64;
	int initial_state_indices_i64;
	int final_state_indices_i64;
	int indexed_state_pool;

	ptrdiff_t out_s0, out_s1, out_s2;
	ptrdiff_t state_s0, state_s1, state_s2;
	ptrdiff_t final_s0, final_s1, final_s2;
	ptrdiff_t qkv_s0, qkv_s1, qkv_s2;
	ptrdiff_t weight_s0, weight_s1, weight_s2;
	ptrdiff_t bias_s0;

	void *out;
	void *conv_state;
	void *final_conv_state; /* nullable when final_state_indices set */
	const void *qkv;
	const void *weight;
	const void *bias; /* nullable */
	const void *cu_seqlens; /* nullable */
	const void *initial_state_indices; /* nullable */
	const void *final_state_indices; /* nullable */
	void *stream; /* mcStream_t */
} InfiniLMCausalConv1dLaunchArgs;

/* Returns 0 on success, non-zero on error. */
int infinilm_causal_conv1d_metax_launch(const InfiniLMCausalConv1dLaunchArgs *args);

#ifdef __cplusplus
}
#endif
