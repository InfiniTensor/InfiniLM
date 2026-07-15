#pragma once

// Hybrid MLP is constructed in MiniCPM5MoeDecoderLayer (register dense or sparse as "mlp")
// so HF weight prefixes stay `mlp.gate_proj` / `mlp.experts.*` without an extra nest.

#include "minicpm5_moe_dense_mlp.hpp"
#include "minicpm5_moe_sparse_moe_block.hpp"
