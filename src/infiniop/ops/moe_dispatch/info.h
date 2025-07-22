#ifndef __MOE_DISPATCH_INFO_H__
#define __MOE_DISPATCH_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <iostream>
#include <vector>

namespace op::moe_dispatch {

class MoEDispatchInfo {
  private:
    int _num_tokens, _k, _hidden_dim, _num_experts;
    infiniDtype_t _data_type;
    infiniDtype_t _index_type;

    MoEDispatchInfo(int num_tokens, int k, int hidden_dim, int num_experts,
                    infiniDtype_t data_type, infiniDtype_t index_type)
        : _num_tokens(num_tokens), _k(k), _hidden_dim(hidden_dim),
          _num_experts(num_experts), _data_type(data_type),
          _index_type(index_type) {}

  public:
    int num_tokens() const { return _num_tokens; }
    int k() const { return _k; }
    int hidden_dim() const { return _hidden_dim; }
    int num_experts() const { return _num_experts; }
    infiniDtype_t data_type() const { return _data_type; }
    infiniDtype_t index_type() const { return _index_type; }

    static utils::Result<MoEDispatchInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t indices_desc,
           infiniopTensorDescriptor_t permuted_output_desc,
           infiniopTensorDescriptor_t aux_info_desc, int num_experts);
};

} // namespace op::moe_dispatch

#endif // __MOE_DISPATCH_INFO_H__ 