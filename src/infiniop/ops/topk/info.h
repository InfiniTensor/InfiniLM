#ifndef __TOPK_INFO_H__
#define __TOPK_INFO_H__

#include "../../../utils/result.hpp"
#include "infiniband/verbs.h"
#include "infinicore.h"
#include "infiniop/tensor_descriptor.h"

namespace op::topk {

enum TopKStrategy {
    DEEPSEEK_V3,
    STANDARD_SOFTMAX
};

class TopKInfo {
  private:
    int _num_tokens, _num_experts, _k;
    infiniDtype_t _data_type;
    size_t _workspace_size;
    TopKStrategy _strategy;
    // New parameters for DeepseekV3
    int _n_group;
    int _topk_group;

    TopKInfo(int num_tokens, int num_experts, int k, infiniDtype_t data_type,
             size_t workspace_size, TopKStrategy strategy, int n_group,
             int topk_group)
        : _num_tokens(num_tokens), _num_experts(num_experts), _k(k),
          _data_type(data_type), _workspace_size(workspace_size),
          _strategy(strategy), _n_group(n_group), _topk_group(topk_group) {}

  public:
    int num_tokens() const { return _num_tokens; }
    int num_experts() const { return _num_experts; }
    int k() const { return _k; }
    infiniDtype_t data_type() const { return _data_type; }
    size_t workspace_size() const { return _workspace_size; }
    TopKStrategy strategy() const { return _strategy; }
    int n_group() const { return _n_group; }
    int topk_group() const { return _topk_group; }

    static utils::Result<TopKInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t output_val_desc,
           infiniopTensorDescriptor_t output_ind_desc,
           infiniopTensorDescriptor_t bias_desc, int k_val,
           TopKStrategy strategy, int n_group, int topk_group);
};

} // namespace op::topk

#endif // __TOPK_INFO_H__ 