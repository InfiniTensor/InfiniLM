#pragma once

namespace infinilm::global_state::decode_phase_profile {

inline bool enabled() { return false; }
inline bool recording() { return false; }
inline double monotonic_ms() { return 0.0; }

struct Counters {
    double attn_ms{0};
    double moe_ms{0};
    double dense_mlp_ms{0};
    double other_layer_ms{0};
    double eager_forward_ms{0};
};

inline Counters &counters() {
    static Counters c;
    return c;
}

inline void reset() {}
inline bool &decode_step_active() {
    static bool active = false;
    return active;
}

} // namespace infinilm::global_state::decode_phase_profile
