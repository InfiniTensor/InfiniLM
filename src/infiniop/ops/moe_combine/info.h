#ifndef __MOE_COMBINE_INFO_H__
#define __MOE_COMBINE_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <iostream>
#include <vector>

namespace op::moe_combine {

class MoECombineInfo {
  private:
    int _num_tokens, _k, _hidden_dim;
    infiniDtype_t _data_type;

    MoECombineInfo(int num_tokens, int k, int hidden_dim,
                   infiniDtype_t data_type)
        : _num_tokens(num_tokens), _k(k), _hidden_dim(hidden_dim),
          _data_type(data_type) {}

  public:
    int num_tokens() const { return _num_tokens; }
    int k() const { return _k; }
    int hidden_dim() const { return _hidden_dim; }
    infiniDtype_t data_type() const { return _data_type; }

    static utils::Result<MoECombineInfo>
    create(infiniopTensorDescriptor_t permuted_input_desc,
           infiniopTensorDescriptor_t gating_weights_desc,
           infiniopTensorDescriptor_t aux_info_desc,
           infiniopTensorDescriptor_t output_desc);
};

} // namespace op::moe_combine

#endif // __MOE_COMBINE_INFO_H__ 