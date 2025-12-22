#ifndef MINICPMV_REF_POS_EMBED_HPP
#define MINICPMV_REF_POS_EMBED_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace minicpmv::ref_pos_embed {

// Matches minicpmv_config/resampler.py:
// - omega = 1 / 10000 ** (i / (D/2))
// - for 2D: meshgrid(w, h) then emb_h uses grid[0] (w), emb_w uses grid[1] (h)
inline void compute_2d_sincos_pos_embed(float *out, size_t embed_dim, size_t h, size_t w) {
    const size_t half = embed_dim / 2;
    const size_t quarter = half / 2;

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            float *dst = out + (y * w + x) * embed_dim;
            for (size_t i = 0; i < quarter; ++i) {
                const float omega = std::pow(10000.0f, -static_cast<float>(i) / static_cast<float>(quarter));
                const float a = static_cast<float>(x) * omega; // w first
                const float b = static_cast<float>(y) * omega; // then h
                dst[i] = std::sin(a);
                dst[i + quarter] = std::cos(a);
                dst[i + half] = std::sin(b);
                dst[i + half + quarter] = std::cos(b);
            }
        }
    }
}

inline std::vector<float> make_2d_sincos_pos_embed(size_t embed_dim, size_t h, size_t w) {
    std::vector<float> out(h * w * embed_dim);
    compute_2d_sincos_pos_embed(out.data(), embed_dim, h, w);
    return out;
}

} // namespace minicpmv::ref_pos_embed

#endif

