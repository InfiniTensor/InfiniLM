#pragma once
#include <cstddef>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct InfiniLMMRoPELaunchArgs {
	int dtype; /* 0=f32,1=f16,2=bf16 */
	int pos_i64;
	int num_tokens;
	int num_q_heads;
	int num_kv_heads;
	int head_size;
	int rotary_dim;
	int section_t;
	int section_h;
	int section_w;
	int interleaved;
	int positions_has_axes;
	int max_position_embeddings;

	ptrdiff_t q_out_s_token, q_out_s_head;
	ptrdiff_t k_out_s_token, k_out_s_head;
	ptrdiff_t q_s_token, q_s_head;
	ptrdiff_t k_s_token, k_s_head;
	ptrdiff_t cos_s_pos, sin_s_pos;
	ptrdiff_t pos_s_axis, pos_s_token;

	void *q_out;
	void *k_out;
	const void *q;
	const void *k;
	const void *cos;
	const void *sin;
	const void *positions;
	void *stream;
} InfiniLMMRoPELaunchArgs;

int infinilm_mrope_metax_launch(const InfiniLMMRoPELaunchArgs *args);
#ifdef __cplusplus
}
#endif
