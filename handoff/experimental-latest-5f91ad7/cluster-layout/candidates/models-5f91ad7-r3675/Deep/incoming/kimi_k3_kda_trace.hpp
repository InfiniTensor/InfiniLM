#pragma once
#include "runtime/runtime.hpp"
#include "global_state/forward_context.hpp"
#include <memory>
namespace infinilm::models::kimi_k3 {
struct KdaTraceImpl;
// Copies snapshots on the owner stream and reads them only after the complete
// KDA call. This avoids inserting a host synchronization between KDA operators.
class KdaTraceBatch {
public:
 KdaTraceBatch(runtime::Runtime&,const global_state::ForwardContext&,size_t);
 ~KdaTraceBatch();
 KdaTraceBatch(const KdaTraceBatch&)=delete;
 KdaTraceBatch&operator=(const KdaTraceBatch&)=delete;
 void capture(const runtime::Tensor&,const char*);
 void finish();
private:
 std::unique_ptr<KdaTraceImpl>impl_;KdaTraceImpl*previous_=nullptr;
};
void trace_kda_projection(runtime::Runtime&,const runtime::Tensor&,const char*);
}
