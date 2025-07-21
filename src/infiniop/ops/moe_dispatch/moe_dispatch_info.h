#ifndef __MOE_DISPATCH_INFO_H__
#define __MOE_DISPATCH_INFO_H__

#include "core/common.h"
#include "core/info.h"
#include "core/tensor_descriptor.h"
#include "utils/data_type.h"
#include <iostream>
#include <vector>

namespace infiniop {

class MoEDispatchInfo : public Info {
  private:
    int num_tokens, k, hidden_dim, num_experts;
    DataType data_type;
    DataType index_type;

  public:
    MoEDispatchInfo(int num_experts,
                    infiniopTensorDescriptor_t input_desc,
                    infiniopTensorDescriptor_t indices_desc,
                    infiniopTensorDescriptor_t permuted_output_desc,
                    infiniopTensorDescriptor_t aux_info_desc) {
        this->num_experts = num_experts;
        this->data_type = input_desc->dtype;
        this->index_type = indices_desc->dtype;
        auto &in_dims = input_desc->dims;
        this->num_tokens = in_dims[0];
        this->hidden_dim = in_dims[1];
        this->k = indices_desc->dims[1];
    }

    void op(void *permuted_output, void *aux_info, const void *input,
            const void *indices, cudaStream_t stream) const;

    void print() const override {
        std::cout << "MoE Dispatch operator, num_tokens: " << num_tokens
                  << ", k: " << k << ", hidden_dim: " << hidden_dim << std::endl;
    }
};

void moe_dispatch_kernel_launcher(const void *input, const void *indices,
                                  void *permuted_output, void *aux_info,
                                  int num_tokens, int k, int hidden_dim, int num_experts,
                                  DataType data_type, DataType index_type,
                                  cudaStream_t stream);

} // namespace infiniop

#endif // __MOE_DISPATCH_INFO_H__ 