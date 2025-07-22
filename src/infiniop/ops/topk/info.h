#ifndef __TOPK_INFO_H__
#define __TOPK_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <iostream>
#include <vector>

namespace op::topk {

class TopKInfo {
  private:
    int _num_tokens, _num_experts, _k;
    infiniDtype_t _data_type;
    size_t _workspace_size;

    TopKInfo(int num_tokens, int num_experts, int k, infiniDtype_t data_type,
             size_t workspace_size)
        : _num_tokens(num_tokens), _num_experts(num_experts), _k(k),
          _data_type(data_type), _workspace_size(workspace_size) {}

  public:
    int num_tokens() const { return _num_tokens; }
    int num_experts() const { return _num_experts; }
    int k() const { return _k; }
    infiniDtype_t data_type() const { return _data_type; }
    size_t workspace_size() const { return _workspace_size; }

    static utils::Result<TopKInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t output_val_desc,
           infiniopTensorDescriptor_t output_ind_desc, int k_val);
};

} // namespace op::topk

#endif // __TOPK_INFO_H__ 