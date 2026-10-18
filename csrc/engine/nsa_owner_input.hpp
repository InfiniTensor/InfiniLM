#pragma once
// NSA-only copied HOST values. No tensors, callbacks, registry or device evidence.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
namespace infinilm::runtime {
class Runtime;
}
namespace infinilm::engine::nsa {
using Page = std::pair<std::uint64_t, std::uint64_t>; // physical, allocation generation
struct Row {
    std::uint64_t incarnation{}, revision{}, past{}, total{};
    std::vector<Page> pages;
    std::vector<std::int64_t> slots;
};
struct Upload {
    std::vector<std::int32_t> past, total, offsets, cu, tables;
    std::size_t table_width{};
    std::vector<std::int64_t> slots;
    bool operator==(const Upload &u) const {
        return past == u.past && total == u.total && offsets == u.offsets
            && cu == u.cu && tables == u.tables && table_width == u.table_width && slots == u.slots;
    }
};
struct Snapshot {
    std::uint64_t pool_epoch{}, page_size{}, page_count{};
    std::vector<Row> rows;
    Upload upload;
};
struct Proof {
    Snapshot owner; // actual effective CPU upload, validated on real worker
    infinilm::runtime::Runtime *runtime{};
    std::uint64_t worker{}, invocation{};
    bool one_tail{};
};
inline bool validate(const Snapshot &s, const Upload &u) {
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    const auto n = s.rows.size();
    if (!s.pool_epoch || !s.page_count || !s.page_size || s.page_size % 64
        || s.page_count > limit || s.page_size > limit || !n || n > s.page_count
        || !u.table_width || u.table_width > s.page_count
        || n > std::numeric_limits<std::size_t>::max() / u.table_width
        || u.past.size() != n || u.total.size() != n || u.tables.size() != n * u.table_width
        || u.offsets.size() != n + 1 || u.cu.size() != n + 1
        || u.offsets.front() != 0 || u.cu.front() != 0) {
        throw std::invalid_argument("NSA owner/upload geometry invalid");
    }
    std::set<std::uint64_t> incarnations, batch_pages;
    std::set<std::int64_t> writers;
    std::uint64_t token_count = 0, total_count = 0;
    bool one_tail = true;
    for (std::size_t i = 0; i < n; ++i) {
        const auto &r = s.rows[i];
        if (!r.incarnation || !r.revision || !incarnations.insert(r.incarnation).second
            || r.total <= r.past || r.total > limit || r.total > s.page_size * s.page_count
            || u.past[i] < 0 || u.total[i] < 0
            || static_cast<std::uint64_t>(u.past[i]) != r.past
            || static_cast<std::uint64_t>(u.total[i]) != r.total) {
            throw std::invalid_argument("NSA request owner/length mismatch");
        }
        const auto count = (r.total - 1) / s.page_size + 1;
        if (r.pages.size() != count || count > u.table_width || r.slots.size() != r.total - r.past) {
            throw std::invalid_argument("NSA active page/slot coverage missing");
        }
        std::set<std::uint64_t> pages;
        for (std::size_t j = 0; j < r.pages.size(); ++j) {
            const auto &p = r.pages[j];
            if (p.first >= s.page_count || !p.second || !pages.insert(p.first).second
                || u.tables[i * u.table_width + j] < 0
                || static_cast<std::uint64_t>(u.tables[i * u.table_width + j]) != p.first) {
                throw std::invalid_argument("NSA active page identity mismatch");
            }
        }
        for (std::size_t j = 0; j < u.table_width; ++j) {
            if (u.tables[i * u.table_width + j] < -1
                || (u.tables[i * u.table_width + j] >= 0
                    && static_cast<std::uint64_t>(u.tables[i * u.table_width + j]) >= s.page_count)) {
                throw std::invalid_argument("NSA uploaded padding value invalid");
            }
        }
        for (std::size_t j = 0; j < r.slots.size(); ++j) {
            const auto logical = r.past + j;
            const auto slot = r.pages[logical / s.page_size].first * s.page_size + logical % s.page_size;
            if (r.slots[j] < 0 || static_cast<std::uint64_t>(r.slots[j]) != slot
                || !writers.insert(r.slots[j]).second
                || token_count + j >= u.slots.size() || u.slots[token_count + j] != r.slots[j]) {
                throw std::invalid_argument("NSA submitted raw slot mismatch/overlap");
            }
        }
        token_count += r.total - r.past;
        total_count += r.total;
        if (token_count > limit || total_count > limit || u.offsets[i + 1] != static_cast<std::int32_t>(token_count)
            || u.cu[i + 1] != static_cast<std::int32_t>(total_count)) {
            throw std::invalid_argument("NSA submitted offsets/cu mismatch");
        }
        one_tail = one_tail && r.total == r.past + 1;
    }
    if (u.slots.size() != token_count) {
        throw std::invalid_argument("NSA slot upload size mismatch");
    }
    // Explicit first-slice restriction: no shared physical prefix in one-tail
    // batch, even when LAST-only could share it. No COW/deduplication claim.
    if (one_tail) {
        for (const auto &r : s.rows) {
            for (const auto &p : r.pages) {
                if (!batch_pages.insert(p.first).second) {
                    throw std::invalid_argument("NSA initial decode domain excludes shared physical prefix pages");
                }
            }
        }
    }
    return one_tail;
}
} // namespace infinilm::engine::nsa
