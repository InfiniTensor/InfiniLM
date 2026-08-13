#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class FloatPower {
public:
    // ==========================================================
    // Dispatcher Schemas
    // ==========================================================

    // Output = Input ^ Scalar (scalar must be double!)
    using schema_scalar = void (*)(Tensor output,
                                   Tensor input,
                                   double exponent);

    // Output = Input ^ Tensor
    using schema_tensor = void (*)(Tensor output,
                                   Tensor input,
                                   Tensor exponent);

    // ==========================================================
    // Execute Entry Points (called by functional interface)
    // ==========================================================

    static void execute(Tensor output,
                        Tensor input,
                        double exponent);

    static void execute(Tensor output,
                        Tensor input,
                        Tensor exponent);

    // ==========================================================
    // Dispatchers
    // ==========================================================

    static common::OpDispatcher<schema_scalar> &dispatcher_scalar();
    static common::OpDispatcher<schema_tensor> &dispatcher_tensor();
};

// =======================================================================
// Functional Interface (Python-visible semantics)
// =======================================================================

// -------------------------------
// 1. Scalar Exponent
// -------------------------------

// out-of-place: ALWAYS float64
Tensor float_power(Tensor input, double exponent);

// in-place
void float_power_(Tensor output, Tensor input, double exponent);

// -------------------------------
// 2. Tensor Exponent
// -------------------------------

// out-of-place: ALWAYS float64
Tensor float_power(Tensor input, Tensor exponent);

// in-place
void float_power_(Tensor output, Tensor input, Tensor exponent);

} // namespace infinicore::op
