#ifndef __TOPK_INFO_H__
#define __TOPK_INFO_H__

#include "core/common.h"
#include "core/info.h"
#include "core/tensor_descriptor.h"
#include "utils/data_type.h"
#include <iostream>
#include <vector>

namespace infiniop {

class TopKInfo : public Info {
  private:
    int num_tokens, num_experts, k;
    DataType data_type;

  public:
    void *workspace;
    size_t workspace_size;
    TopKInfo(infiniopTensorDescriptor_t input_desc,
             infiniopTensorDescriptor_t output_val_desc,
             infiniopTensorDescriptor_t output_ind_desc, int k_val) {
        this->k = k_val;
        this->data_type = input_desc->dtype;
        auto &in_dims = input_desc->dims;
        this->num_tokens = in_dims[0];
        this->num_experts = in_dims[1];

        // For now, workspace size is determined by the logic from top_k.cu
        const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
	    const bool needs_workspace = !is_pow_2 || num_experts > 256;
	    this->workspace_size = needs_workspace ? num_tokens * num_experts * sizeOf(data_type) : 0;
    }

    void op(void *output_val, void *output_ind, const void *input,
            cudaStream_t stream) const;

    void print() const override {
        std::cout << "TopK operator, num_tokens: " << num_tokens
                  << ", num_experts: " << num_experts << ", k: " << k
                  << ", data_type: " << (int)data_type << std::endl;
    }
};

void topk_kernel_launcher(const void *input, void *output_val, void *output_ind,
                            void *workspace, int num_tokens, int num_experts,
                            int k, DataType data_type, cudaStream_t stream);

} // namespace infiniop

#endif // __TOPK_INFO_H__ 