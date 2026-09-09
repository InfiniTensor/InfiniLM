/*
 * This file contains some code derived from xtensor (https://github.com/xtensor-stack/xtensor),
 * which is licensed under the BSD-3-Clause license.
 */

#include "infinicore/io.hpp"
#include "../utils/custom_types.h"
#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Tensor = infinicore::Tensor;
using TensorSliceParams = infinicore::TensorSliceParams;
using DataType = infinicore::DataType;
using Device = infinicore::Device;
using TensorImpl = infinicore::TensorImpl;
using Size = infinicore::Size;

/**
 * @brief This function extracts a scalar or sub-tensor from a tensor using a vector of indexes.
 */
inline Tensor at_impl(const Tensor &tensor, const std::vector<Size> &indexes) {
    if (indexes.size() > tensor->ndim()) {
        throw std::runtime_error("at_impl:: Number of indexes (" + std::to_string(indexes.size()) + ") exceeds tensor dimensions (" + std::to_string(tensor->ndim()) + ")");
    }

    for (size_t i = 0; i < indexes.size(); i++) {
        if (indexes[i] >= tensor->shape()[i]) {
            throw std::runtime_error("at_impl :: Index " + std::to_string(indexes[i]) + " is out of bounds for dimension " + std::to_string(i));
        }
    }

    std::vector<TensorSliceParams> slices;
    slices.reserve(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        slices.push_back({i, indexes[i], 1});
    }

    Tensor result = tensor->narrow(slices);
    for (size_t i = 0; i < indexes.size(); i++) {
        result = result->squeeze(0);
    }

    return result;
}

template <typename... Args>
Tensor at(const Tensor &tensor, Args... args) {
    std::vector<Size> indexes = {static_cast<Size>(args)...};
    return at_impl(tensor, indexes);
}

[[maybe_unused]] Tensor at(const Tensor &tensor, std::initializer_list<Size> indexes) {
    std::vector<Size> indexes_vec(indexes.begin(), indexes.end());
    return at_impl(tensor, indexes_vec);
}

Tensor at(const Tensor &tensor, const std::vector<Size> &indexes) {
    return at_impl(tensor, indexes);
}

/**
 * @brief read a value from raw data pointer based on DataType.
 */
template <typename T>
T item_impl(const std::byte *data, DataType dtype) {
    switch (dtype) {
    case DataType::kFloat16: {
        const fp16_t *ptr = reinterpret_cast<const fp16_t *>(data);
        float f = _f16_to_f32(ptr[0]);
        return static_cast<T>(f);
    }
    case DataType::kBFloat16: {
        const bf16_t *ptr = reinterpret_cast<const bf16_t *>(data);
        float f = _bf16_to_f32(ptr[0]);
        return static_cast<T>(f);
    }
    default:
        break;
    }

    const T *ptr = reinterpret_cast<const T *>(data);
    return ptr[0];
}

/**
 * @brief Extracts a scalar value from a single-element tensor.
 * The tensor must have exactly one element and must be located on CPU device.
 *
 * @code{.cpp}
 * float value = item<float>(tensor, dtype);  // Extract as float
 * @endcode
 */
template <typename T>
T item(const Tensor &tensor) {
    if (tensor->numel() != 1) {
        throw std::runtime_error("item() can only be called on a tensor with exactly one element, but got " + std::to_string(tensor->numel()) + " elements");
    }

    if (tensor->device().type() != Device::Type::kCpu) {
        throw std::runtime_error("item() can only be called on a CPU tensor, but got device: " + tensor->device().ToString());
    }

    const std::byte *data = tensor->data();
    DataType dtype = tensor->dtype();
    return item_impl<T>(data, dtype);
}
} // namespace

namespace infinicore {
namespace print_options {

template <class S>
class fmtflags_guard {
public:
    explicit fmtflags_guard(S &stream)
        : m_stream(stream), m_flags(stream.flags()) {}
    ~fmtflags_guard() { m_stream.flags(m_flags); }

private:
    S &m_stream;
    std::ios_base::fmtflags m_flags;
};

struct PrintOptionsImpl {
    int edge_items = 3;   // default edge items: 3 means print 3 items of each dimension.
    int line_width = 80;  // default line width: 75 means print 75 chars per line.
    int threshold = 1000; // default threshold: 1000 means print 1000 elements of the tensor.
    int precision = 4;    // default precision: -1 means no precision limit.
    int sci_mode = -1;    // default sci_mode: -1 means auto decision.
};

inline PrintOptionsImpl &print_options() {
    static PrintOptionsImpl po;
    return po;
}

void set_line_width(int line_width) {
    print_options().line_width = line_width;
}

void set_threshold(int threshold) {
    print_options().threshold = threshold;
}

void set_edge_items(int edge_items) {
    print_options().edge_items = edge_items;
}

void set_precision(int precision) {
    print_options().precision = precision;
}

void set_sci_mode(int sci_mode) {
    print_options().sci_mode = sci_mode;
}

/**
 * @brief read print options from the out stream and global settings.
 */
inline print_options::PrintOptionsImpl get_print_options(std::ostream &out) {
    print_options::PrintOptionsImpl res;

// Macro to read option from stream, apply default if not set, or reset stream value
#define PROCESS_PRINT_OPTION(OPTION)                                       \
    res.OPTION = static_cast<int>(out.iword(print_options::OPTION::id())); \
    if (res.OPTION > 0) {                                                  \
        out.iword(print_options::OPTION::id()) = long(-1);                 \
    } else {                                                               \
        res.OPTION = print_options::print_options().OPTION;                \
    }

    // Process all print options
    PROCESS_PRINT_OPTION(edge_items);
    PROCESS_PRINT_OPTION(line_width);
    PROCESS_PRINT_OPTION(threshold);
    PROCESS_PRINT_OPTION(precision);

    res.sci_mode = print_options::print_options().sci_mode;

#undef PROCESS_PRINT_OPTION
    return res;
}

template <class T, class E = void>
struct Printer;

/**
 * @brief Printer specialization for floating-point types (float, double, long double).
 */
template <class T>
struct Printer<T, std::enable_if_t<std::is_floating_point<T>::value>> {
    using value_type = T;
    using cache_type = std::vector<value_type>;
    using cache_iterator = typename cache_type::const_iterator;

    explicit Printer(std::streamsize precision, int sci_mode = 0) : m_precision(precision), m_sci_mode(sci_mode) {}

    void calculate() {
        m_precision = m_precision > m_required_precision ? m_required_precision : m_precision;
        m_it = m_cache.cbegin();

        if (m_scientific) {
            // 3 = sign, number and dot and 4 = "e+00"
            m_width = m_precision + 7;
            if (m_large_exponent) {
                // = e+000 (additional number)
                m_width += 1;
            }
        } else {
            std::streamsize decimals = 1; // print a leading 0
            if (std::floor(m_max) != 0) {
                decimals += std::streamsize(std::log10(std::floor(m_max)));
            }
            // 2 => sign and dot
            m_width = 2 + decimals + m_precision;
        }
        if (!m_required_precision) {
            --m_width;
        }
    }

    std::ostream &print_next(std::ostream &out) {
        if ((1 == m_sci_mode) || ((-1 == m_sci_mode) && m_scientific)) {
            if (!m_large_exponent) {
                out << std::scientific;
                out.width(m_width);
                out << (*m_it);
            } else {
                std::stringstream buf;
                buf.width(m_width);
                buf << std::scientific;
                buf.precision(m_precision);
                buf << (*m_it);
                std::string res = buf.str();

                if (res[res.size() - 4] == 'e') {
                    res.erase(0, 1);
                    res.insert(res.size() - 2, "0");
                }
                out << res;
            }
        } else {
            std::stringstream buf;
            buf.width(m_width);
            buf << std::fixed;
            buf.precision(m_precision);
            buf << (*m_it);
            if (!m_required_precision && !std::isinf(*m_it) && !std::isnan(*m_it)) {
                buf << '.';
            }
            std::string res = buf.str();
            auto sit = res.rbegin();
            while (*sit == '0') {
                *sit = ' ';
                ++sit;
            }
            out << res;
        }
        ++m_it;
        return out;
    }

    void update(const value_type &val) {
        if (val != 0 && !std::isinf(val) && !std::isnan(val)) {
            if (!m_scientific || !m_large_exponent) {
                int exponent = 1 + int(std::log10(std::abs(val)));
                if (exponent <= -5 || exponent > 7) {
                    m_scientific = true;
                    m_required_precision = m_precision;
                    if (exponent <= -100 || exponent >= 100) {
                        m_large_exponent = true;
                    }
                }
            }

            if (std::abs(val) > m_max) {
                m_max = std::abs(val);
            }
            if (m_required_precision < m_precision) {
                while (std::floor(val * std::pow(10, m_required_precision)) != val * std::pow(10, m_required_precision)) {
                    m_required_precision++;
                }
            }
        }
        m_cache.push_back(val);
    }

    std::streamsize width() const { return m_width; }

private:
    bool m_large_exponent = false;
    bool m_scientific = false;

    std::streamsize m_width = 9;
    std::streamsize m_precision;
    std::streamsize m_required_precision = 0;
    value_type m_max = 0;
    int m_sci_mode = -1;
    cache_type m_cache;
    cache_iterator m_it;
};

/**
 * @brief Printer specialization for integer types (signed and unsigned integers).
 */
template <class T>
struct Printer<
    T, std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value>> {
    using value_type = T;
    using cache_type = std::vector<value_type>;
    using cache_iterator = typename cache_type::const_iterator;

    explicit Printer(std::streamsize, int sci_mode = 0) {}

    void calculate() {
        m_it = m_cache.cbegin();
        m_width = 1 + std::streamsize((m_max > 0) ? std::log10(m_max) : 0) + m_sign;
    }

    std::ostream &print_next(std::ostream &out) {
        // + enables printing of chars etc. as numbers
        // TODO should chars be printed as numbers?
        out.width(m_width);
        out << +(*m_it);
        ++m_it;
        return out;
    }

    void update(const value_type &val) {
        // For unsigned types, abs is not needed (always non-negative)
        // For signed types, we need to take absolute value
        value_type abs_val;
        if constexpr (std::is_signed<value_type>::value) {
            abs_val = (val < 0) ? -val : val;
        } else {
            abs_val = val;
        }

        if (abs_val > m_max) {
            m_max = abs_val;
        }

        if (std::is_signed<value_type>::value && val < 0) {
            m_sign = true;
        }
        m_cache.push_back(val);
    }

    std::streamsize width() { return m_width; }

private:
    std::streamsize m_width;
    bool m_sign = false;
    value_type m_max = 0;

    cache_type m_cache;
    cache_iterator m_it;
};

/**
 * @brief Printer specialization for bool type.
 */
template <class T>
struct Printer<
    T, std::enable_if_t<std::is_same<T, bool>::value>> {
    using value_type = bool;
    using cache_type = std::vector<bool>;
    using cache_iterator = typename cache_type::const_iterator;

    explicit Printer(std::streamsize, int sci_mode = 0) {}

    void calculate() {
        m_it = m_cache.cbegin();
    }

    std::ostream &print_next(std::ostream &out) {
        if (*m_it) {
            out << " true";
        } else {
            out << "false";
        }
        // TODO: the following std::setw(5) isn't working correctly on OSX.
        // out << std::boolalpha << std::setw(m_width) << (*m_it);
        ++m_it;
        return out;
    }

    void update(const value_type &val) { m_cache.push_back(val); }

    std::streamsize width() { return m_width; }

private:
    std::streamsize m_width = 5;
    cache_type m_cache;
    cache_iterator m_it;
};

} // namespace print_options
} // namespace infinicore

namespace infinicore {
namespace print_options {

/**
 * @brief Recursively traverses tensor dimensions to collect values for printing.
 */
template <class T>
void recurser_run(Printer<T> &printer,
                  const Tensor &tensor,
                  std::vector<Size> indexes,
                  std::size_t lim = 0) {

    using size_type = Size;
    const auto view = at(tensor, indexes);
    if (view->ndim() == 0) {
        T value = item<T>(view);
        printer.update(value);
    } else {
        size_type i = 0;
        for (; i != static_cast<size_type>(view->shape()[0] - 1); ++i) {
            if (lim && size_type(view->shape()[0]) > (lim * 2) && i == lim) {
                i = static_cast<size_type>(view->shape()[0]) - lim;
                if (lim <= 1) {
                    break;
                }
            }
            indexes.push_back(static_cast<int>(i));
            recurser_run(printer, tensor, indexes, lim);
            indexes.pop_back();
        }
        indexes.push_back(static_cast<int>(i));
        recurser_run(printer, tensor, indexes, lim);
        indexes.pop_back();
    }
}

/**
 * @brief Recursively prints tensor elements with proper formatting.
 */
template <class T>
std::ostream &xoutput(std::ostream &out,
                      const Tensor &tensor,
                      std::vector<size_t> &indexes,
                      Printer<T> &printer,
                      std::size_t blanks,
                      std::streamsize element_width,
                      std::size_t edge_items,
                      std::size_t line_width) {

    using size_type = Size;
    const auto view = at(tensor, indexes);
    if (view->ndim() == 0) {
        printer.print_next(out);
    } else {
        std::string indents(blanks, ' ');

        size_type i = 0;
        size_type elems_on_line = 0;
        const size_type ewp2 = static_cast<size_type>(element_width) + size_type(2);
        const size_type line_lim = static_cast<size_type>(std::floor(line_width / ewp2));

        out << '[';
        for (; i != size_type(view->shape()[0] - 1); ++i) {

            if (edge_items && size_type(view->shape()[0]) > (edge_items * 2) && i == edge_items) {
                if (view->ndim() == 1 && line_lim != 0 && elems_on_line >= line_lim) {
                    out << " ...,";
                } else if (view->ndim() > 1) {
                    elems_on_line = 0;
                    out << "...," << std::endl
                        << indents;
                } else {
                    out << "..., ";
                }
                i = size_type(view->shape()[0]) - edge_items;
                if (edge_items <= 1) {
                    break;
                }
            }
            if (view->ndim() == 1 && line_lim != 0 && elems_on_line >= line_lim) {
                out << std::endl
                    << indents;
                elems_on_line = 0;
            }

            indexes.push_back(static_cast<int>(i));
            xoutput(out, tensor, indexes, printer, blanks + 1, element_width, edge_items,
                    line_width)
                << ',';
            indexes.pop_back();
            elems_on_line++;

            if ((view->ndim() == 1) && !(line_lim != 0 && elems_on_line >= line_lim)) {
                ; // out << ' ';
            } else if (view->ndim() > 1) {
                out << std::endl
                    << indents;
            }
        }
        if (view->ndim() == 1 && line_lim != 0 && elems_on_line >= line_lim) {
            out << std::endl
                << indents;
        }

        indexes.push_back(static_cast<int>(i));
        xoutput(out, tensor, indexes, printer, blanks + 1, element_width, edge_items,
                line_width)
            << ']';
        indexes.pop_back();
    }
    return out;
}

template <class T>
std::ostream &pretty_print(const Tensor &original_tensor,
                           std::ostream &out = std::cout) {
    Tensor tensor = original_tensor->to(Device::Type::kCpu);
    bool on_cpu = original_tensor->device() == Device::Type::kCpu;
    std::string device_str = original_tensor->device().ToString();
    infinicore::context::syncDevice();

    fmtflags_guard<std::ostream> guard(out);

    std::size_t edge_items = 0;
    Size sz = tensor->numel();
    auto po = get_print_options(out);

    if (sz > static_cast<std::size_t>(po.threshold)) {
        edge_items = static_cast<std::size_t>(po.edge_items);
    }
    if (sz == 0) {
        out << "[]";
        return out;
    }

    auto temp_precision = out.precision();
    auto precision = temp_precision;

    if (po.precision != -1) {
        out.precision(static_cast<std::streamsize>(po.precision));
        precision = static_cast<std::streamsize>(po.precision);
    }

    Printer<T> printer(precision, po.sci_mode);
    std::vector<size_t> indexes = {};

    recurser_run(printer, tensor, indexes, edge_items);

    printer.calculate();
    indexes.clear();

    auto element_width = printer.width();

    out << "tensor(";
    xoutput(out,
            tensor,
            indexes,
            printer,
            1 + 7,
            element_width,
            edge_items,
            static_cast<std::size_t>(po.line_width));

    if (!on_cpu) {
        out << ", device=" << '\'' << device_str << '\'';
    }

    out << ", dtype=infinicore." << toString(tensor->dtype()) << ")\n";
    out.precision(temp_precision); // restore precision
    return out;
}

} // namespace print_options
} // namespace infinicore

namespace infinicore {
std::ostream &operator<<(std::ostream &out, const Tensor &tensor) {
    if (!tensor) {
        out << "tensor([])\n";
        return out;
    }

    switch (tensor->dtype()) {
    case DataType::kInt8: // 3
    {
        return infinicore::print_options::pretty_print<int8_t>(tensor, out);
    }
    case DataType::kInt16: // 4
    {
        return infinicore::print_options::pretty_print<int16_t>(tensor, out);
    }
    case DataType::kInt32: // 5
    {
        return infinicore::print_options::pretty_print<int32_t>(tensor, out);
    }
    case DataType::kInt64: // 6
    {
        return infinicore::print_options::pretty_print<int64_t>(tensor, out);
    }
    case DataType::kUInt8: // 7
    {
        return infinicore::print_options::pretty_print<uint8_t>(tensor, out);
    }
    case DataType::kUInt16: // 8
    {
        return infinicore::print_options::pretty_print<uint16_t>(tensor, out);
    }
    case DataType::kUInt32: // 9
    {
        return infinicore::print_options::pretty_print<uint32_t>(tensor, out);
    }
    case DataType::kUInt64: // 10
    {
        return infinicore::print_options::pretty_print<uint64_t>(tensor, out);
    }
    case DataType::kFloat16: // 12
    {
        // Convert F16 to F32 for printing
        return infinicore::print_options::pretty_print<float>(tensor, out);
    }
    case DataType::kFloat32: // 13
    {
        return infinicore::print_options::pretty_print<float>(tensor, out);
    }
    case DataType::kFloat64: // 14
    {
        return infinicore::print_options::pretty_print<double>(tensor, out);
    }
    case DataType::kBFloat16: // 19
    {
        // Convert BF16 to F32 for printing
        return infinicore::print_options::pretty_print<float>(tensor, out);
    }
    default:
        throw std::runtime_error("cant not print unknown dtype tensor : " + toString(tensor->dtype()));
    }

    return out;
}
} // namespace infinicore
