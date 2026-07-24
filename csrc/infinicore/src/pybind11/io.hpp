#pragma once

#include "infinicore.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::io {

inline void bind(py::module &m) {
    m.def(
        "set_printoptions", [](int precision, int threshold, int edge_items, int line_width, py::object sci_mode) {
            infinicore::print_options::set_precision(precision);
            infinicore::print_options::set_threshold(threshold);
            infinicore::print_options::set_edge_items(edge_items);
            infinicore::print_options::set_line_width(line_width);

            // Handle sci_mode: None -> -1 (auto), True -> 1 (enable), False -> 0 (disable)
            int sci_mode_value = -1; // default: auto
            if (!sci_mode.is_none()) {
                sci_mode_value = static_cast<int>(py::cast<bool>(sci_mode)); // True -> 1, False -> 0
            }

            infinicore::print_options::set_sci_mode(sci_mode_value); }, pybind11::arg("precision"), pybind11::arg("threshold"), pybind11::arg("edge_items"), pybind11::arg("line_width"), pybind11::arg("sci_mode"));
}

} // namespace infinicore::io
