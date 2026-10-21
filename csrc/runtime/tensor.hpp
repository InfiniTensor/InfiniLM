#pragma once

#include "device.hpp"
#include "dtype.hpp"
#include "memory.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

#include <infini/rt.h>
namespace infinilm::runtime {

using Size = std::size_t;
using Stride = std::ptrdiff_t;
using Shape = std::vector<Size>;
using Strides = std::vector<Stride>;

class TensorImpl;
class Runtime;

struct TensorMetaData {
    Shape shape;
    Strides strides;
    DataType dtype;

    TensorMetaData(const Shape &shape, const Strides &strides, const DataType &dtype);
};

struct TensorData {
    size_t offset;
    std::shared_ptr<Memory> memory;
};

struct TensorSliceParams {
    size_t dim;
    size_t start;
    Size len;
};

class Tensor {
public:
    static Tensor empty(Runtime &runtime, const Shape &shape, const DataType &dtype,
                        bool pin_memory = false);

    static Tensor empty(const Shape &shape,
                        const DataType &dtype,
                        const Device &device,
                        bool pin_memory = false);

    static Tensor empty_parameter(Runtime &runtime,
                                  const Shape &shape,
                                  const DataType &dtype);

    // Exact-sized backing for long-lived weights and caches, not Tensor views.
    static Tensor empty_exact(Runtime &runtime,
                              const Shape &shape,
                              const DataType &dtype);

    static Tensor strided_empty(Runtime &runtime,
                                const Shape &shape,
                                const Strides &strides,
                                const DataType &dtype,
                                bool pin_memory = false);

    static Tensor ones(const Shape &shape,
                       const DataType &dtype,
                       const Device &device,
                       bool pin_memory = false);

    static Tensor from_blob(void *raw_ptr,
                            const Shape &shape,
                            const DataType &dtype,
                            const Device &device);

    static Tensor from_blob(void *raw_ptr,
                            const Shape &shape,
                            const DataType &dtype,
                            const Device &device,
                            std::shared_ptr<void> owner);

    static Tensor strided_from_blob(void *raw_ptr,
                                    const Shape &shape,
                                    const Strides &strides,
                                    const DataType &dtype,
                                    const Device &device);

    static Tensor strided_from_blob(void *raw_ptr,
                                    const Shape &shape,
                                    const Strides &strides,
                                    const DataType &dtype,
                                    const Device &device,
                                    std::shared_ptr<void> owner);

    void reset() noexcept {
        impl_.reset();
    }

    void reset(std::shared_ptr<TensorImpl> new_impl) noexcept {
        impl_ = std::move(new_impl);
    }

    bool empty() const noexcept {
        return impl_ == nullptr;
    }

    size_t use_count() const noexcept {
        return impl_ ? impl_.use_count() : 0;
    }

    Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) = default;

    TensorImpl *operator->();
    const TensorImpl *operator->() const;

    operator bool() const;

protected:
    Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
    friend class TensorImpl;
    friend std::ostream &operator<<(std::ostream &out, const Tensor &tensor);
};

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {

public:
    TensorImpl(const Shape &shape, const DataType &dtype);
    TensorImpl(const Shape &shape, const Strides &strides, const DataType &dtype);

    std::byte *data();
    const std::byte *data() const;

    const Shape &shape() const;

    const Strides &strides() const;

    bool is_contiguous() const;

    Size ndim() const;

    Size numel() const;

    Size size(size_t dim) const;

    size_t element_size() const;

    size_t nbytes() const;

    Stride stride(size_t dim) const;

    DataType dtype() const;

    Device device() const;

    infini::rt::TensorView view();

    bool is_pinned() const;
    std::shared_ptr<Runtime> allocation_owner() const { return data_.memory->allocation_owner(); }

    std::string info() const;

    /**
     * Unsafe API that returns a new tensor with the same raw memory untracked by allocator
     * This API is used for loosely tracking a piece of memory while allowing it to be reused,
     * typically in a compute graph scenario.
     */
    Tensor to_blob_(Runtime &runtime) const;

    /**
     * Unsafe API that returns a new tensor with the same memory and let allocator retracks the memory.
     * Should only be used on the tensor returned by to_blob_(runtime).
     */
    Tensor resume_from_blob_(Runtime &runtime) const;

    ///
    /// Data Transfer APIs
    ///

    /**
     * Returns a new tensor with the same data on a different device.
     * If the new device passed is same as the current device, the original tensor is returned.
     *
     * @param device The device of the new tensor
     *
     * @return A new tensor with the same data on the specified device
     */
    Tensor to(Device device) const;
    // Source-owned, completed GPU -> CPU transfer; no caller TLS selection.
    Tensor to(Runtime &runtime, Device device) const;
    Tensor to(Runtime &source_runtime, Runtime &destination_runtime) const;

    /**
     * Copy Data from another tensor to this tensor.
     *
     * @param src The source tensor to copy from
     *
     * @return A new tensor with the same data on the specified device
     */
    void copy_from(Tensor src);

    // Copy on the owning Runtime without consulting caller TLS. The
    // destination belongs to runtime for same-device/H2D; for D2H the GPU
    // source belongs to runtime and destination is CPU. Host data is consumed
    // or produced before this call returns. D2H is refused while recording.
    void copy_from(Runtime &runtime, Tensor src);
    // Explicit completed-source handoff. Allocation identity is not a last writer;
    // external/other-stream writers must be completed by the caller before entry.
    void copy_from(Runtime &destination_runtime, Runtime &source_runtime, Tensor src);

    /**
     * Return a tensor with the same data in contiguous arrangement as current tensor.
     * If this tensor is already contiguous, the original tensor is returned.
     *
     * @return A new tensor with the same data on the specified device
     */
    Tensor contiguous() const;
    Tensor contiguous(Runtime &runtime) const;

    ///
    /// View APIs
    ///

    /**
     * Returns a new tensor with a dimension of size one removed at the specified position.
     * Throws runtime_error if the dimension to be removed is not of size 1.
     *
     * @param dim The dimension index to remove
     * @return A new tensor with the removed dimension
     *
     * Example:
     *   // For a 3D tensor with shape [1, 3, 4], squeeze at dim 0 results in shape [3, 4]
     *   tensor->squeeze(0);
     */
    Tensor squeeze(size_t dim) const;

    /**
     * Returns a new tensor with a dimension of size one inserted at the specified position.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param dim The dimension index at which to insert the new dimension
     * @return A new tensor with the added dimension
     *
     * Example:
     *   // For a 2D tensor with shape [3, 4], unsqueeze at dim 0 results in shape [1, 3, 4]
     *   // unsqueeze at dim 1 results in shape [3, 1, 4]
     *   // unsqueeze at dim 2 results in shape [3, 4, 1]
     *   tensor->unsqueeze(0);
     */
    Tensor unsqueeze(size_t dim) const;

    /**
     * Returns a new tensor that is a narrowed version of the current tensor.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param slices A vector of slice parameters specifying the dimension, start index,
     *               and length for each dimension to narrow
     * @return A new tensor with narrowed dimensions
     *
     * Example:
     *   // Narrow dimension 0 from index 2 to 5 (length 3)
     *   // and dimension 1 from index 1 to 3 (length 2)
     *   tensor.narrow({{0, 2, 3}, {1, 1, 2}});
     */
    Tensor narrow(const std::vector<TensorSliceParams> &slices) const;

    /**
     * Returns a new tensor with the dimensions permuted (reordered) according to the given order.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param order The desired ordering of dimensions
     * @return A new tensor with permuted dimensions
     *
     * Example:
     *   // For a 3D tensor with shape [2, 3, 4], permute to [2, 0, 1]
     *   // This swaps the dimensions: dim0->dim2, dim1->dim0, dim2->dim1
     *   tensor->permute({2, 0, 1});
     */
    Tensor permute(const Shape &order) const;

    /**
     * Returns a new tensor with the same data but a different shape.
     * The returned tensor shares the same underlying storage with the original tensor.
     * The tensor is rearranged if the new shape is not compatible with the current shape.
     *
     * @param new_shape The desired new shape
     * @return A new tensor with the specified shape
     *
     * Example:
     *   // Reshape a 2x3 tensor (6 elements) to a 3x2 tensor
     *   tensor->view({3, 2});
     */
    Tensor view(const Shape &new_shape) const;

    /**
     * Insecurely returns a new tensor with the specified shape and strides.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param new_shape The desired new shape
     * @param new_strides The desired new strides
     * @return A new tensor with the specified shape and strides
     *
     * Example:
     *   // Create a non-contiguous view with custom strides
     *   tensor->as_strided({2, 3}, {6, 2}); // Stride of 6 for dim0, 2 for dim1
     */
    Tensor as_strided(const Shape &new_shape, const Strides &new_strides) const;

protected:
    static size_t storage_span_bytes(
        const Shape &shape,
        const Strides &strides,
        size_t element_size);

    friend class Tensor;

protected:
    TensorMetaData meta_;
    TensorData data_;

private:
    // Mark to indicate if the tensor is created from to_blob_()
    bool to_blob_mark_ = false;
};

} // namespace infinilm::runtime
