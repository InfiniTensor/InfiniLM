#ifndef __MOE_COMBINE_INFO_H__
#define __MOE_COMBINE_INFO_H__

#include "core/common.h"
#include "core/info.h"
#include "core/tensor_descriptor.h"
#include "utils/data_type.h"
#include <iostream>
#include <vector>

namespace infiniop {

class MoECombineInfo : public Info {
  private:
    int num_tokens, k, hidden_dim;
    DataType data_type;

  public:
    MoECombineInfo(infiniopTensorDescriptor_t permuted_input_desc,
                   infiniopTensorDescriptor_t gating_weights_desc,
                   infiniopTensorDescriptor_t aux_info_desc,
                   infiniopTensorDescriptor_t output_desc) {
        this->data_type = output_desc->dtype;
        auto &out_dims = output_desc->dims;
        this->num_tokens = out_dims[0];
        this->hidden_dim = out_dims[1];
        this->k = gating_weights_desc->dims[1];
    }

    void op(void *output, const void *permuted_input,
            const void *gating_weights, const void *aux_info,
            cudaStream_t stream) const;

    void print() const override {
        std::cout << "MoE Combine operator, num_tokens: " << num_tokens
                  << ", k: " << k << ", hidden_dim: " << hidden_dim << std::endl;
    }
};

void moe_combine_kernel_launcher(const void *permuted_input,
                                 const void *gating_weights,
                                 const void *aux_info, void *output,
                                 int num_tokens, int k, int hidden_dim,
                                 DataType data_type, cudaStream_t stream);

} // namespace infiniop

#endif // __MOE_COMBINE_INFO_H__ 