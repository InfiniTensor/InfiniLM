#ifndef INFER_TENSOR_H
#define INFER_TENSOR_H

#include "infinicore_infer.h"
#include "utils.hpp"
#include <memory>
#include <string>
#include <vector>

struct Storage {
    void *memory;
    size_t size;
    infiniDevice_t device_type;
    int device_id;

    static std::shared_ptr<Storage> create(size_t size);
    static std::shared_ptr<Storage> createAsync(size_t size, infinirtStream_t stream = nullptr);
    static std::shared_ptr<Storage> createHost(size_t size);
    ~Storage();
};

struct SliceParams {
    size_t dim;
    size_t start;
    size_t len;
};

class TensorDesc {
private:
    infiniopTensorDescriptor_t _desc;

public:
    static std::shared_ptr<TensorDesc>
    create(infiniDtype_t dtype, const std::vector<size_t> &shape,
           const std::vector<ptrdiff_t> &strides);
    infiniopTensorDescriptor_t get() const { return _desc; };
    ~TensorDesc();
};

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    infiniDtype_t _dtype;
    std::vector<size_t> _shape;
    std::vector<ptrdiff_t> _strides;
    void *_data;
    ptrdiff_t _offset;
    size_t _size;
    std::shared_ptr<Storage> storage;
    infiniopTensorDescriptor_t _desc;

    void *data_impl(ptrdiff_t offset) const;
    std::shared_ptr<Tensor>
    slice_impl(const std::vector<SliceParams> &slices) const;

public:
    static std::shared_ptr<Tensor> buffer(infiniDtype_t dtype,
                                          const std::vector<size_t> &shape,
                                          infinirtStream_t stream = nullptr);
    static std::shared_ptr<Tensor> weight(void *host_data,
                                          infiniDtype_t dtype,
                                          const std::vector<size_t> &shape);
    std::shared_ptr<Tensor> slice(size_t dim, size_t start, size_t len);
    std::shared_ptr<Tensor const> slice(size_t dim, size_t start,
                                        size_t len) const;
    std::shared_ptr<Tensor> slice(const std::vector<SliceParams> &slices);
    std::shared_ptr<Tensor const>
    slice(const std::vector<SliceParams> &slices) const;
    std::shared_ptr<Tensor> dim_merge(size_t dim_start, size_t dim_end);
    std::shared_ptr<Tensor> dim_split(size_t dim,
                                      const std::vector<size_t> &dims);
    std::shared_ptr<Tensor> permute(const std::vector<size_t> &order);
    void *data(ptrdiff_t offset = 0);
    void const *data(ptrdiff_t offset = 0) const;
    void copy_from(std::shared_ptr<Tensor const> src, infiniopHandle_t handle,
                   infinirtStream_t stream = nullptr);
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    size_t ndim() const;
    infiniDtype_t dtype() const;
    std::shared_ptr<TensorDesc> desc() const;
    size_t byte_size() const;
    ptrdiff_t data_offset() const;
    infiniDevice_t device_type() const;
    int device_id() const;
    bool is_contigous() const;

    void debug(const std::string &filename) const;
    void debug() const;

    ~Tensor();
};

inline size_t dsize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_INVALID:
        return 0;
    case INFINI_DTYPE_BYTE:
        return 1;
    case INFINI_DTYPE_BOOL:
        return 1;
    case INFINI_DTYPE_I8:
        return 1;
    case INFINI_DTYPE_I16:
        return 2;
    case INFINI_DTYPE_I32:
        return 4;
    case INFINI_DTYPE_I64:
        return 8;
    case INFINI_DTYPE_U8:
        return 1;
    case INFINI_DTYPE_U16:
        return 2;
    case INFINI_DTYPE_U32:
        return 4;
    case INFINI_DTYPE_U64:
        return 8;
    case INFINI_DTYPE_F8:
        return 1;
    case INFINI_DTYPE_F16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    case INFINI_DTYPE_F64:
        return 8;
    case INFINI_DTYPE_C16:
        return 2;
    case INFINI_DTYPE_C32:
        return 4;
    case INFINI_DTYPE_C64:
        return 8;
    case INFINI_DTYPE_C128:
        return 16;
    case INFINI_DTYPE_BF16:
        return 2;
    default:
        return 0;
    }
}

#endif
