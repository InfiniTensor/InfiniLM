#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

namespace vllm {

class ScalarType {
public:
    enum NanRepr : uint8_t {
        NAN_NONE = 0,
        NAN_IEEE_754 = 1,
        NAN_EXTD_RANGE_MAX_MIN = 2,

        NAN_REPR_ID_MAX
    };

    constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                         int32_t bias, bool finite_values_only = false,
                         NanRepr nan_repr = NAN_IEEE_754)
        : exponent(exponent),
          mantissa(mantissa),
          signed_(signed_),
          bias(bias),
          finite_values_only(finite_values_only),
          nan_repr(nan_repr) {}

    static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
        return ScalarType(0, size_bits - 1, true, bias);
    }

    static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
        return ScalarType(0, size_bits, false, bias);
    }

    static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                              uint8_t mantissa) {
        return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
    }

    static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                       bool finite_values_only,
                                       NanRepr nan_repr) {
        return ScalarType(exponent, mantissa, true, 0,
                          finite_values_only, nan_repr);
    }

    static inline ScalarType float_checked(uint8_t exponent,
                                           uint8_t mantissa,
                                           bool finite_values_only,
                                           NanRepr nan_repr) {
        if (!(nan_repr < NAN_REPR_ID_MAX)) {
            throw std::runtime_error("Invalid NanRepr");
        }

        if (!(mantissa > 0 && exponent > 0)) {
            throw std::runtime_error("mantissa/exponent must > 0");
        }

        if (nan_repr == NAN_IEEE_754) {
            throw std::runtime_error("use float_IEEE754");
        }

        return float_(exponent, mantissa, finite_values_only, nan_repr);
    }

    uint8_t const exponent;
    uint8_t const mantissa;
    bool const signed_;
    int32_t const bias;

    bool const finite_values_only;
    NanRepr const nan_repr;

    using Id = int64_t;

private:
    template <typename T_>
    static constexpr size_t member_id_field_width() {
        using T = std::decay_t<T_>;
        return std::is_same<T, bool>::value ? 1 : sizeof(T) * 8;
    }

    template <typename Fn, typename Init, typename Member, typename... Rest>
    static constexpr auto reduce_members_helper(Fn f, Init val, Member member,
                                                Rest... rest) {
        auto new_val = f(val, member);
        if constexpr (sizeof...(rest) > 0) {
            return reduce_members_helper(f, new_val, rest...);
        } else {
            return new_val;
        }
    }

    template <typename Fn, typename Init>
    constexpr auto reduce_members(Fn f, Init init) const {
        return reduce_members_helper(f, init, exponent, mantissa, signed_, bias,
                                     finite_values_only, nan_repr);
    }

    template <typename Fn, typename Init>
    static constexpr auto reduce_member_types(Fn f, Init init) {
        constexpr auto dummy = ScalarType(0, 0, false, 0, false, NAN_NONE);
        return dummy.reduce_members(f, init);
    }

    static constexpr auto id_size_bits() {
        return reduce_member_types(
            [](int acc, auto member) -> int {
                return acc + member_id_field_width<decltype(member)>();
            },
            0);
    }

public:
    constexpr Id id() const {
        static_assert(id_size_bits() <= sizeof(Id) * 8,
                      "ScalarType id too large");

        auto fn = [](std::pair<Id, uint32_t> result, auto member) {
            auto [id, offset] = result;
            constexpr auto bits = member_id_field_width<decltype(member)>();
            return std::pair<Id, uint32_t>{
                id | (int64_t(member) & ((uint64_t(1) << bits) - 1)) << offset,
                offset + bits};
        };

        return reduce_members(fn, std::pair<Id, uint32_t>{}).first;
    }

    static constexpr ScalarType from_id(Id id) {
        auto fn = [id](auto result, auto member) {
            using T = decltype(member);
            auto [tuple, offset] = result;
            constexpr auto bits = member_id_field_width<T>();
            auto val = static_cast<T>((id >> offset) & ((uint64_t(1) << bits) - 1));
            return std::pair{std::tuple_cat(tuple, std::make_tuple(val)), offset + bits};
        };

        auto [args, _] = reduce_member_types(fn, std::pair<std::tuple<>, int>{});

        return std::apply([](auto... xs) { return ScalarType(xs...); }, args);
    }

    constexpr int64_t size_bits() const {
        return mantissa + exponent + (signed_ ? 1 : 0);
    }

    constexpr bool is_signed() const { return signed_; }
    constexpr bool is_integer() const { return exponent == 0; }
    constexpr bool is_floating_point() const { return exponent > 0; }

    constexpr bool is_ieee_754() const {
        return is_floating_point() && !finite_values_only && nan_repr == NAN_IEEE_754;
    }

    constexpr bool has_nans() const {
        return is_floating_point() && nan_repr != NAN_NONE;
    }

    constexpr bool has_infs() const {
        return is_floating_point() && !finite_values_only;
    }

    constexpr bool has_bias() const { return bias != 0; }

    std::string str() const {
        if (is_floating_point()) {
            auto ret = "float" + std::to_string(size_bits()) + "_e" + std::to_string(exponent) + "m" + std::to_string(mantissa);

            if (!is_ieee_754()) {
                if (finite_values_only) {
                    ret += "f";
                }
                if (nan_repr != NAN_NONE) {
                    ret += "n";
                }
            }
            return ret;
        } else {
            auto ret = (signed_ ? "int" : "uint") + std::to_string(size_bits());
            if (has_bias()) {
                ret += "b" + std::to_string(bias);
            }
            return ret;
        }
    }

    constexpr bool operator==(ScalarType const &other) const {
        return mantissa == other.mantissa && exponent == other.exponent && bias == other.bias && signed_ == other.signed_ && finite_values_only == other.finite_values_only && nan_repr == other.nan_repr;
    }
};

using ScalarTypeId = ScalarType::Id;

static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);

static inline constexpr auto kFE2M1f = ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE3M2f = ScalarType::float_(3, 2, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn = ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu = ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);

static inline constexpr auto kInt4 = kS4;
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;
static inline constexpr auto kInt8 = kS8;
static inline constexpr auto kUint8 = kU8;
static inline constexpr auto kUint8b128 = kU8B128;

static inline constexpr auto kFloat4_e2m1f = kFE2M1f;
static inline constexpr auto kFloat6_e3m2f = kFE3M2f;
static inline constexpr auto kFloat8_e4m3fn = kFE4M3fn;
static inline constexpr auto kFloat8_e5m2 = kFE5M2;
static inline constexpr auto kFloat16_e8m7 = kFE8M7;
static inline constexpr auto kFloat16_e5m10 = kFE5M10;

static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

static inline constexpr auto kFloat16Id = kFloat16.id();

} // namespace vllm
