#pragma once

namespace infini::ops::native_flash {

// This intentionally names only the head dimensions qualified by this native
// provider. The pinned FA 2.8.3 kernels round 160 and 224 to the next 64.
inline int head_dim_bin(int dimension) {
  switch (dimension) {
    case 32: case 64: case 96: case 128:
    case 192: case 256: return dimension;
    case 160: return 192;
    case 224: return 256;
    default: return 0;
  }
}

inline int split_block_n(int bin) {
  return bin <= 64 ? 256 : (bin <= 128 ? 128 : 64);
}

}  // namespace infini::ops::native_flash
