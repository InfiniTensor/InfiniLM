#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InfiniLMFusedGatingLaunchArgs {
	int dtype; /* 0=f32, 1=f16, 2=bf16 for A_log/a/b/dt_bias */
	int batch;
	int seq_len;
	int hidden;
	float beta;
	float threshold;

	ptrdiff_t g_s0, g_s1, g_s2;
	ptrdiff_t beta_s0, beta_s1, beta_s2;
	ptrdiff_t A_log_s0;
	ptrdiff_t a_s0, a_s1, a_s2;
	ptrdiff_t b_s0, b_s1, b_s2;
	ptrdiff_t dt_bias_s0;

	float *g; /* always f32 */
	float *beta_output; /* always f32 */
	const void *A_log;
	const void *a;
	const void *b;
	const void *dt_bias;
	void *stream;
} InfiniLMFusedGatingLaunchArgs;

int infinilm_fused_gated_delta_net_gating_metax_launch(const InfiniLMFusedGatingLaunchArgs *args);

#ifdef __cplusplus
}
#endif
