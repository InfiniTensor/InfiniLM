#include "../tensor.hpp"
#include "infinicore/ops/float_power.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

using infinicore::Tensor;
using infinicore::op::float_power;
using infinicore::op::float_power_;

inline Tensor unwrap(py::handle obj) {
    try {
        return obj.cast<Tensor>();
    } catch (...) {}

    if (py::hasattr(obj, "_underlying")) {
        return obj.attr("_underlying").cast<Tensor>();
    }

    throw py::type_error("Expected infinicore.Tensor, but got " + py::repr(obj.get_type()).cast<std::string>());
}

void bind_float_power(py::module &m) {

    // --- Out-of-place: float_power(input, exponent) ---
    m.def(
        "float_power", [](py::object input_obj, py::object exp_obj) -> Tensor {
            Tensor input = unwrap(input_obj);

            // 处理标量指数的情况 (float 或 int)
            if (py::isinstance<py::float_>(exp_obj) || py::isinstance<py::int_>(exp_obj)) {
                return float_power(input, exp_obj.cast<double>());
            }

            // 处理张量指数的情况
            Tensor exponent = unwrap(exp_obj);
            return float_power(input, exponent);
        },
        py::arg("input"), py::arg("exponent"));

    // --- In-place: float_power_(out, input, exponent) ---
    m.def(
        "float_power_", [](py::object out_obj, py::object input_obj, py::object exp_obj) {
            Tensor out = unwrap(out_obj);
            Tensor input = unwrap(input_obj);

            if (py::isinstance<py::float_>(exp_obj) || py::isinstance<py::int_>(exp_obj)) {
                float_power_(out, input, exp_obj.cast<double>());
            } else {
                Tensor exponent = unwrap(exp_obj);
                float_power_(out, input, exponent);
            }
        },
        py::arg("out"), py::arg("input"), py::arg("exponent"));
}

} // namespace infinicore::ops
