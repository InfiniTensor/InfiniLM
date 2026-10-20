#pragma once

namespace infini::ops::native_flash {

// The pinned FA 2.8.3 launchers specialize these rounded widths. Kernels keep
// the logical d and use Is_even_K=false masks for shorter vector-aligned rows.
inline int head_dim_bin(int dimension) {
  if (dimension <= 0 || dimension > 256 || dimension % 8) return 0;
  if (dimension <= 128) return ((dimension + 31) / 32) * 32;
  return ((dimension + 63) / 64) * 64;
}

inline int split_block_n(int bin) {
  return bin <= 64 ? 256 : (bin <= 128 ? 128 : 64);
}

}  // namespace infini::ops::native_flash
