#include "infinicore.hpp"
#include <cstdint>
#include <limits>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace py = pybind11;

namespace infinicore::tensor {

namespace {

constexpr const char *kElementTypeError = "List elements must be bool, int, or float";

// Zero-copy view over list / tuple / generic sequence; items() exposes the
// underlying PyObject** for hot loop indexing.

class SequenceView {
public:
    enum class Kind : uint8_t { List,
                                Tuple,
                                Fast };

    explicit SequenceView(py::handle obj, const char *err = "Could not convert object to sequence")
        : SequenceView(obj.ptr(), err) {}

    explicit SequenceView(PyObject *obj, const char *err = "Could not convert object to sequence") {
        if (PyList_Check(obj)) {
            kind_ = Kind::List;
            borrowed_ = obj;
            fast_ = nullptr;
            return;
        }
        if (PyTuple_Check(obj)) {
            kind_ = Kind::Tuple;
            borrowed_ = obj;
            fast_ = nullptr;
            return;
        }
        kind_ = Kind::Fast;
        borrowed_ = nullptr;
        fast_ = PySequence_Fast(obj, err);
        if (!fast_) {
            throw py::error_already_set();
        }
    }

    ~SequenceView() {
        if (fast_ != nullptr) {
            Py_DECREF(fast_);
        }
    }

    SequenceView(const SequenceView &) = delete;
    SequenceView &operator=(const SequenceView &) = delete;

    bool is_list() const {
        return kind_ == Kind::List;
    }

    Py_ssize_t size() const {
        return size_of(ptr());
    }

    PyObject *item(Py_ssize_t index) const {
        return item_at(ptr(), index);
    }

    // Direct pointer to ob_item; valid for list / tuple / PySequence_Fast result.
    PyObject **items() const {
        return PySequence_Fast_ITEMS(ptr());
    }

    static Py_ssize_t size_of(PyObject *obj) {
        if (PyList_Check(obj)) {
            return PyList_GET_SIZE(obj);
        }
        if (PyTuple_Check(obj)) {
            return PyTuple_GET_SIZE(obj);
        }
        return PySequence_Size(obj);
    }

    static PyObject *item_at(PyObject *obj, Py_ssize_t index) {
        if (PyList_Check(obj)) {
            return PyList_GET_ITEM(obj, index);
        }
        if (PyTuple_Check(obj)) {
            return PyTuple_GET_ITEM(obj, index);
        }
        return PySequence_GetItem(obj, index);
    }

private:
    PyObject *ptr() const {
        return borrowed_ != nullptr ? borrowed_ : fast_;
    }

    Kind kind_ = Kind::Fast;
    PyObject *borrowed_ = nullptr;
    PyObject *fast_ = nullptr;
};

// Shape discovery + uniform per-scalar traversal.

inline bool is_python_scalar(PyObject *obj) {
    return PyBool_Check(obj) || PyFloat_Check(obj) || PyLong_Check(obj);
}

inline bool is_python_sequence(PyObject *obj) {
    return PyList_Check(obj) || PyTuple_Check(obj);
}

struct ListLayout {
    SequenceView seq;
    int ndim = 0;
    Shape shape;
    bool list_of_lists = false;

    explicit ListLayout(py::handle obj)
        : seq(obj) {
        if (!is_python_sequence(obj.ptr())) {
            throw py::type_error("Input data must be a list or tuple");
        }
        if (seq.size() == 0) {
            throw py::value_error("Input data cannot be empty");
        }

        PyObject *first = seq.item(0);
        if (is_python_scalar(first)) {
            ndim = 1;
            shape = Shape{static_cast<Size>(seq.size())};
            return;
        }

        if (!is_python_sequence(first)) {
            throw py::type_error("List elements must be scalars or nested lists");
        }
        if (SequenceView::size_of(first) == 0) {
            throw py::value_error("Nested list cannot be empty");
        }
        if (!is_python_scalar(SequenceView::item_at(first, 0))) {
            throw py::value_error("Only 1D and 2D lists are supported");
        }

        ndim = 2;
        const Size rows = static_cast<Size>(seq.size());
        const Size cols = static_cast<Size>(SequenceView::size_of(first));
        list_of_lists = seq.is_list() && PyList_Check(first);

        for (Py_ssize_t i = 1; i < seq.size(); ++i) {
            PyObject *row_obj = seq.item(i);
            if (!is_python_sequence(row_obj)) {
                throw py::value_error("Input must be a regular 2D list with equal row lengths");
            }
            if (static_cast<Size>(SequenceView::size_of(row_obj)) != cols) {
                throw py::value_error("Input must be a regular 2D list with equal row lengths");
            }
            if (list_of_lists && !PyList_Check(row_obj)) {
                list_of_lists = false;
            }
        }

        shape = Shape{rows, cols};
    }
};

template <typename Fn>
void for_each_scalar(const ListLayout &layout, Fn &&fn) {
    const Py_ssize_t rows = layout.seq.size();
    PyObject **rows_items = layout.seq.items();

    if (layout.ndim == 1) {
        for (Py_ssize_t i = 0; i < rows; ++i) {
            fn(rows_items[i]);
        }
        return;
    }

    const Py_ssize_t cols = static_cast<Py_ssize_t>(layout.shape[1]);
    if (layout.list_of_lists) {
        for (Py_ssize_t i = 0; i < rows; ++i) {
            PyObject **row_items = PySequence_Fast_ITEMS(rows_items[i]);
            for (Py_ssize_t j = 0; j < cols; ++j) {
                fn(row_items[j]);
            }
        }
        return;
    }

    for (Py_ssize_t i = 0; i < rows; ++i) {
        PyObject *row = rows_items[i];
        for (Py_ssize_t j = 0; j < cols; ++j) {
            fn(SequenceView::item_at(row, j));
        }
    }
}

// read_pylong_fast inlines compact int reads on Python >= 3.12 via
// PyUnstable_Long_*; falls back to PyLong_AsLongLong otherwise.
inline int64_t read_pylong_fast(PyObject *obj) {
#if PY_VERSION_HEX >= 0x030C0000
    PyLongObject *lo = reinterpret_cast<PyLongObject *>(obj);
    if (PyUnstable_Long_IsCompact(lo)) {
        return static_cast<int64_t>(PyUnstable_Long_CompactValue(lo));
    }
#endif
    const int64_t value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return value;
}

inline uint64_t read_pyuint64(PyObject *obj) {
    const auto value = PyLong_AsUnsignedLongLong(obj);
    if (value == std::numeric_limits<unsigned long long>::max() && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return static_cast<uint64_t>(value);
}

// Convert int64 -> Dst with range check for narrow ints (i8/i16/i32,
// u8/u16/u32) and uint64 (forbids negatives); bool/i64/float/double truncate.
template <typename Dst>
inline Dst narrow_int_to(int64_t v) {
    if constexpr (std::is_floating_point_v<Dst>) {
        return static_cast<Dst>(v);
    } else if constexpr (std::is_same_v<Dst, bool>) {
        return v != 0;
    } else if constexpr (std::is_same_v<Dst, int64_t>) {
        return v;
    } else if constexpr (std::is_same_v<Dst, uint64_t>) {
        if (v < 0) {
            throw std::overflow_error("Integer value out of range for target dtype");
        }
        return static_cast<uint64_t>(v);
    } else if constexpr (std::is_unsigned_v<Dst>) {
        if (v < 0 || static_cast<uint64_t>(v) > std::numeric_limits<Dst>::max()) {
            throw std::overflow_error("Integer value out of range for target dtype");
        }
        return static_cast<Dst>(v);
    } else {
        if (v < std::numeric_limits<Dst>::min() || v > std::numeric_limits<Dst>::max()) {
            throw std::overflow_error("Integer value out of range for target dtype");
        }
        return static_cast<Dst>(v);
    }
}

template <typename Dst>
inline Dst read_pyint(PyObject *obj) {
    if constexpr (std::is_same_v<Dst, uint64_t>) {
        return read_pyuint64(obj);
    } else {
        return narrow_int_to<Dst>(read_pylong_fast(obj));
    }
}

// Write path: branch order is PyLong/PyFloat exact first (most inputs), then
// bool (int subclass), then long/float subclass fallback. F16/BF16 share
// write_dtype_half via float pivot; other dtypes share write_dtype_native.
template <typename Dst>
void write_dtype_native(const ListLayout &layout, std::byte *dest) {
    Dst *out = reinterpret_cast<Dst *>(dest);
    for_each_scalar(layout, [&](PyObject *obj) {
        if (PyLong_CheckExact(obj)) {
            *out++ = read_pyint<Dst>(obj);
        } else if (PyFloat_CheckExact(obj)) {
            *out++ = static_cast<Dst>(PyFloat_AS_DOUBLE(obj));
        } else if (PyBool_Check(obj)) {
            *out++ = static_cast<Dst>(obj == Py_True);
        } else if (PyLong_Check(obj)) {
            *out++ = read_pyint<Dst>(obj);
        } else if (PyFloat_Check(obj)) {
            *out++ = static_cast<Dst>(PyFloat_AS_DOUBLE(obj));
        } else {
            throw py::type_error(kElementTypeError);
        }
    });
}

template <typename Dst>
void write_dtype_half(const ListLayout &layout, std::byte *dest) {
    Dst *out = reinterpret_cast<Dst *>(dest);
    for_each_scalar(layout, [&](PyObject *obj) {
        float f;
        if (PyLong_CheckExact(obj)) {
            f = static_cast<float>(read_pylong_fast(obj));
        } else if (PyFloat_CheckExact(obj)) {
            f = static_cast<float>(PyFloat_AS_DOUBLE(obj));
        } else if (PyBool_Check(obj)) {
            f = (obj == Py_True) ? 1.0f : 0.0f;
        } else if (PyLong_Check(obj)) {
            f = static_cast<float>(read_pylong_fast(obj));
        } else if (PyFloat_Check(obj)) {
            f = static_cast<float>(PyFloat_AS_DOUBLE(obj));
        } else {
            throw py::type_error(kElementTypeError);
        }
        *out++ = Dst::FromFloat(f);
    });
}

void write_with_dtype(const ListLayout &layout, DataType dtype, std::byte *dest) {
    switch (dtype) {
    case DataType::kInt8:
        return write_dtype_native<int8_t>(layout, dest);
    case DataType::kInt16:
        return write_dtype_native<int16_t>(layout, dest);
    case DataType::kInt32:
        return write_dtype_native<int32_t>(layout, dest);
    case DataType::kInt64:
        return write_dtype_native<int64_t>(layout, dest);
    case DataType::kUInt8:
        return write_dtype_native<uint8_t>(layout, dest);
    case DataType::kUInt16:
        return write_dtype_native<uint16_t>(layout, dest);
    case DataType::kUInt32:
        return write_dtype_native<uint32_t>(layout, dest);
    case DataType::kUInt64:
        return write_dtype_native<uint64_t>(layout, dest);
    case DataType::kFloat16:
        return write_dtype_half<infini::rt::Float16>(layout, dest);
    case DataType::kBFloat16:
        return write_dtype_half<infini::rt::BFloat16>(layout, dest);
    case DataType::kFloat32:
        return write_dtype_native<float>(layout, dest);
    case DataType::kFloat64:
        return write_dtype_native<double>(layout, dest);
    default:
        throw py::type_error(
            std::string("Unsupported dtype for from_list: ") + toString(dtype));
    }
}

} // namespace

// Entry (exported for pybind registration in tensor.hpp).
Tensor from_list_py(py::handle data, DataType dtype) {
    const ListLayout layout(data);
    auto tensor = Tensor::empty(layout.shape, dtype, Device(Device::Type::kCpu, 0));
    write_with_dtype(layout, dtype, tensor->data());
    return tensor;
}

} // namespace infinicore::tensor
