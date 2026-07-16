#pragma once

#include "tensor.hpp"
#include <iostream>

namespace infinicore::print_options {

/**
 * @brief Sets the line width. After \a line_width chars, a new line is added.
 * @param line_width The line width
 */
void set_line_width(int line_width);

/**
 * @brief Sets the threshold after which summarization is triggered (default: 1000).
 * @param threshold The number of elements in the tensor that triggers summarization in the output
 */
void set_threshold(int threshold);

/**
 * @brief Sets the number of edge items.
 * If the summarization is triggered, this value defines how many items of each dimension are printed.
 * @param edge_items The number of edge items
 */
void set_edge_items(int edge_items);

/**
 * @brief Sets the precision for printing floating point values.
 * @param precision The number of digits for floating point output
 */

void set_precision(int precision);

/**
 * @brief Sets the sci mode of the floating point values when printing an Tensor.
 * @param sci_mode The sci mode: -1 for auto decision, 0 to disable, 1 to enable
 */

void set_sci_mode(int sci_mode); // -1: auto, 0: disable, 1: enable

#define DEFINE_LOCAL_PRINT_OPTION(NAME)                                 \
    class NAME {                                                        \
    public:                                                             \
        NAME(int value) : m_value(value) { id(); }                      \
        static int id() {                                               \
            static int id = std::ios_base::xalloc();                    \
            return id;                                                  \
        }                                                               \
        int value() const { return m_value; }                           \
                                                                        \
    private:                                                            \
        int m_value;                                                    \
    };                                                                  \
                                                                        \
    inline std::ostream &operator<<(std::ostream &out, const NAME &n) { \
        out.iword(NAME::id()) = n.value();                              \
        return out;                                                     \
    }

/**
 * @class line_width
 * io manipulator used to set the width of the lines when printing an Tensor.
 *
 * @code{.cpp}
 * using po = infinicore::print_options;
 * std::cout << po::line_width(100) << tensor << std::endl;
 * @endcode
 */
DEFINE_LOCAL_PRINT_OPTION(line_width)

/**
 * io manipulator used to set the threshold after which summarization is triggered.
 */
DEFINE_LOCAL_PRINT_OPTION(threshold)

/**
 * io manipulator used to set the number of egde items if the summarization is triggered.
 */
DEFINE_LOCAL_PRINT_OPTION(edge_items)

/**
 * io manipulator used to set the precision of the floating point values when printing an Tensor.
 */
DEFINE_LOCAL_PRINT_OPTION(precision)

} // namespace infinicore::print_options
