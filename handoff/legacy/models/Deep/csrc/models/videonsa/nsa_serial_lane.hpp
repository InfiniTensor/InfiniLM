#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace infinilm::models::videonsa::nsa_serial {
struct Page {
    std::uint64_t physical{}, generation{};
    bool operator==(const Page &p) const {
        return physical == p.physical && generation == p.generation;
    }
};
struct Binding {
    // The model binds tokens to exact immutable device/dtype/origin/shape/strides
    // and retains strong raw K/V + compressed K/V leases outside this helper.
    std::uint64_t pool_epoch{}, worker{}, lane{}, raw{}, compressed{};
    std::uint64_t page_count{}, page_size{}, block_size{};
    bool operator==(const Binding &b) const {
        return pool_epoch == b.pool_epoch && worker == b.worker && lane == b.lane
            && raw == b.raw && compressed == b.compressed && page_count == b.page_count
            && page_size == b.page_size && block_size == b.block_size;
    }
};
struct Row {
    std::uint64_t incarnation{}, revision{}, past{}, total{}, slot{};
    std::vector<Page> pages; // active pages only, not uploaded padding
};
struct Evidence { Binding binding; std::vector<Row> rows; };
enum class Status { Absent, QueuedOnOwnerStream, Completed };
enum class Boundary { HostSubmissionOnly, ExistingOwnerStreamSync };

class Lane {
    const std::thread::id owner_ = std::this_thread::get_id();
    std::optional<Evidence> previous_;
    std::optional<Evidence> current_;
    Status status_{Status::Absent};
    bool enqueued_{};
    void owner() const {
        if (owner_ != std::this_thread::get_id()) throw std::logic_error("foreign NSA worker");
    }
    static void validate(const Evidence &e) {
        const auto &b = e.binding;
        if (!b.pool_epoch || !b.worker || !b.lane || !b.raw || !b.compressed
            || !b.page_count || !b.page_size || !b.block_size || b.page_size % b.block_size
            || b.page_count > std::numeric_limits<std::uint64_t>::max() / b.page_size
            || e.rows.empty()) throw std::invalid_argument("NSA binding/rows missing");
        std::set<std::uint64_t> keys, slots;
        for (const auto &r : e.rows) {
            if (!r.incarnation || !r.revision || !keys.insert(r.incarnation).second
                || r.past == std::numeric_limits<std::uint64_t>::max()
                || r.total != r.past + 1 || r.total > b.page_count * b.page_size
                || r.pages.size() != (r.total - 1) / b.page_size + 1)
                throw std::invalid_argument("NSA one-tail owner evidence invalid");
            std::set<std::uint64_t> physical;
            for (const auto &p : r.pages)
                if (p.physical >= b.page_count || !p.generation || !physical.insert(p.physical).second)
                    throw std::invalid_argument("NSA active page evidence invalid");
            if (r.slot != r.pages.back().physical * b.page_size + r.past % b.page_size
                || !slots.insert(r.slot).second) throw std::invalid_argument("NSA raw writer overlap/slot");
        }
    }
    bool reusable(const Row &r) const {
        const auto &rows = previous_->rows;
        const auto it = std::find_if(rows.begin(), rows.end(), [&](const Row &p) {
            return p.incarnation == r.incarnation;
        });
        return it != rows.end() && it->revision == r.revision && it->total == r.past
            && it->pages.size() <= r.pages.size()
            && std::equal(it->pages.begin(), it->pages.end(), r.pages.begin());
    }
    static void writers(const Evidence &e, bool last) {
        std::set<std::uint64_t> domains;
        const auto per_page = e.binding.page_size / e.binding.block_size;
        for (const auto &r : e.rows) {
            const auto tail = (r.total - 1) / e.binding.block_size;
            for (auto i = last ? tail : 0; i <= tail; ++i) {
                const auto domain = r.pages[i / per_page].physical * per_page + i % per_page;
                if (!domains.insert(domain).second)
                    throw std::invalid_argument("NSA compressed writer overlap before raw write");
            }
        }
    }
public:
    Lane() = default;
    Lane(const Lane &) = delete;
    Lane &operator=(const Lane &) = delete;
    // Call before ANY paged_caching_ write, after actual CPU preflight/leases.
    // One current frame; no tickets, pending map, callback epochs or registry.
    bool begin(Evidence e) {
        owner();
        if (current_) throw std::logic_error("NSA RUN already active");
        validate(e);
        if (previous_ && !(previous_->binding == e.binding)
            && status_ == Status::QueuedOnOwnerStream)
            throw std::logic_error("NSA queued backing replacement needs actual owner drain");
        const bool last = previous_ && previous_->binding == e.binding
            && status_ != Status::Absent
            && std::all_of(e.rows.begin(), e.rows.end(), [&](const Row &r) { return reusable(r); });
        writers(e, last); // rejection leaves current/previous unchanged, zero cache writes
        current_ = std::move(e);
        enqueued_ = false;
        return last;
    }
    void compression_enqueued() {
        owner();
        if (!current_ || enqueued_) throw std::logic_error("NSA enqueue order");
        enqueued_ = true; // ONLY after successful eager native Call return
    }
    // Worker calls for each model-owned NSA lane before job_done. Completed is
    // legal ONLY immediately after its already-existing successful owner sync.
    void finish(Boundary boundary) {
        owner();
        if (!current_ || !enqueued_) throw std::logic_error("NSA no successful submission");
        previous_ = std::move(current_);
        current_.reset();
        status_ = boundary == Boundary::ExistingOwnerStreamSync
            ? Status::Completed : Status::QueuedOnOwnerStream;
        enqueued_ = false;
    }
    // RUN catch, cache reset/rebinding, untracked native writer, prefill/non-NSA
    // cache-writing path and retire-all use this; no late callbacks accepted.
    // Revocation clears CPU validity, NOT permission to release in-flight leases.
    void revoke() {
        owner(); previous_.reset(); current_.reset();
        status_ = Status::Absent; enqueued_ = false;
    }
    Status status() const { owner(); return status_; }
    std::size_t retained_rows() const {
        owner(); return previous_ ? previous_->rows.size() : 0;
    }
};
} // namespace infinilm::models::videonsa::nsa_serial
