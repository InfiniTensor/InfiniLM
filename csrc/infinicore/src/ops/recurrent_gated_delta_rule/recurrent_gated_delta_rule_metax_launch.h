#pragma once
#include <cstddef>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct InfiniLMRecurrentGatedLaunchArgs {
	int dtype;
	int gate_dtype;
	int B;
	int T;
	int Hk;
	int Hv;
	int Dk;
	int Dv;
	int pool_size;
	int value_heads_per_key_head;
	int use_qk_l2norm;
	int indexed_state_pool;
	int initial_state_indices_i64;
	int final_state_indices_i64;
	int has_final_state;

	ptrdiff_t out_s0, out_s1, out_s2;
	ptrdiff_t initial_s0, initial_s1, initial_s2, initial_s3;
	ptrdiff_t final_s0, final_s1, final_s2, final_s3;
	ptrdiff_t q_s0, q_s1, q_s2;
	ptrdiff_t k_s0, k_s1, k_s2;
	ptrdiff_t v_s0, v_s1, v_s2;
	ptrdiff_t g_s0, g_s1, g_s2;
	ptrdiff_t beta_s0, beta_s1, beta_s2;

	void *out;
	void *initial_state;
	void *final_state;
	const void *q;
	const void *k;
	const void *v;
	const void *g;
	const void *beta;
	const void *initial_state_indices;
	const void *final_state_indices;
	void *stream;
} InfiniLMRecurrentGatedLaunchArgs;

int infinilm_recurrent_gated_delta_rule_metax_launch(const InfiniLMRecurrentGatedLaunchArgs *args);

#ifdef __cplusplus
}
#endif
