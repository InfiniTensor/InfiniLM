#pragma once
#include "kimi_owned_indices_raw_r1213.hpp"
extern "C" {
void*modern_kimi_graph_create_r1650(const void*owner,int device,void*stream,void*metadata,size_t metadata_bytes,void*workspace,size_t workspace_bytes,size_t pool,const int64_t*offsets,const int64_t*reads,const int64_t*writes);
int modern_kimi_graph_capture_begin_r1650(void*raw,const void*owner);
int modern_kimi_graph_capture_end_r1650(void*raw,const void*owner);
int modern_kimi_graph_abort_r1650(void*raw,const void*owner);
int modern_kimi_graph_replay_r1650(void*raw,const void*owner);
const char*modern_kimi_graph_info_r1650(void*raw,const void*owner);
int modern_kimi_graph_close_r1650(void*raw,const void*owner);
}
