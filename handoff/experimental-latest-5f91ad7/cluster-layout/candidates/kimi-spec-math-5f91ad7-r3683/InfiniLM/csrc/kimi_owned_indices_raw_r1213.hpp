#pragma once
#include <cstddef>
#include <cstdint>
#include <infini/rt.h>
extern "C" {
const char *modern_kimi_owned_error_r1213();
void *modern_kimi_owned_begin_r1213(int device,void *stream,void *metadata,size_t metadata_bytes,void *workspace,size_t workspace_bytes,size_t requests,size_t tokens,size_t pool,const int64_t *offsets,const int64_t *reads,const int64_t *writes,int mode);
int modern_kimi_owned_end_r1213(void *context);
int modern_kimi_owned_kda_raw_r1213(void *context,const infini::rt::TensorView *values,float scale,float lower,bool normalize);
int modern_kimi_owned_conv_raw_r1213(void *context,const infini::rt::TensorView *values);
int modern_kimi_owned_stats_r1213(int device,uint64_t *out);
}
