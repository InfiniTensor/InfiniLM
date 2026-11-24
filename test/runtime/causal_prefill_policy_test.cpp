#include <cassert>
#include <initializer_list>

#include "csrc/layers/attention/backends/causal_prefill_policy.hpp"

int main() {
    using infinilm::layers::attention::backends::detail::UseFusedCausalPrefill;
    for (const char *flag : {"", "0", "true", "01", "1 "}) {
        assert(!UseFusedCausalPrefill(flag, true, true, true, 3, 32, 2910, 2910));
    }
    assert(!UseFusedCausalPrefill(nullptr, true, true, true, 3, 32, 2910, 2910));
    assert(UseFusedCausalPrefill("1", true, true, true, 3, 32, 2910, 2910));
    assert(UseFusedCausalPrefill("1", true, true, true, 3, 64, 521, 4096));
    assert(!UseFusedCausalPrefill("1", false, true, true, 3, 32, 2910, 2910));
    assert(!UseFusedCausalPrefill("1", true, false, true, 3, 32, 2910, 2910));
    assert(!UseFusedCausalPrefill("1", true, true, false, 3, 32, 2910, 2910));
    assert(!UseFusedCausalPrefill("1", true, true, true, 2, 32, 2910, 2910));
    assert(!UseFusedCausalPrefill("1", true, true, true, 3, 0, 2910, 2910));
    assert(!UseFusedCausalPrefill("1", true, true, true, 3, 32, 1, 4096));
    assert(!UseFusedCausalPrefill("1", true, true, true, 3, 32, 2910, 2909));
    assert(!UseFusedCausalPrefill("1", true, true, true, 3, 32, 2910, 4097));
}
