#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::minicpm5_moe {

/// Dense FFN for shallow layers (first_k_dense_replace). Reuses AOT-ready layers::MLP.
using MiniCPM5DenseMLP = infinilm::layers::MLP;

} // namespace infinilm::models::minicpm5_moe
