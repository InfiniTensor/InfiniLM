#ifndef __INFINIOP_TENSOR_H__
#define __INFINIOP_TENSOR_H__

#include "infiniop/tensor_descriptor.h"
#include <string>
#include <vector>

struct InfiniopTensorDescriptor {
private:
    // Datatype
    infiniDtype_t _dtype;
    // Shape of the tensor
    std::vector<size_t> _shape;
    // Stride of each dimension in elements
    std::vector<ptrdiff_t> _strides;

public:
    InfiniopTensorDescriptor(infiniDtype_t dtype, size_t ndim, const size_t *shape, const ptrdiff_t *strides);
    ~InfiniopTensorDescriptor() = default;
    infiniDtype_t dtype() const;
    std::vector<size_t> shape() const;
    size_t dim(size_t i) const;
    size_t ndim() const;
    std::vector<ptrdiff_t> strides() const;
    ptrdiff_t stride(size_t i) const;
    std::vector<ptrdiff_t> getByteStrides() const;
    bool isContiguous(size_t dim_start, size_t dim_end) const;
    bool isContiguous() const;
    size_t numel() const;

    // a dim is broadcasted if it's corresponding stride is 0 but dim > 1
    bool hasBroadcastDim() const;
    std::vector<size_t> getBroadcastDim() const;

    infiniopTensorDescriptor_t dimMerge(size_t dim_start, size_t dim_end) const;
    infiniopTensorDescriptor_t dimSplit(size_t axis, const std::vector<size_t> &dims) const;
    infiniopTensorDescriptor_t dimPermute(const std::vector<size_t> &order) const;

    std::string toString() const;
};

#endif // __INFINIOP_TENSOR_H__
