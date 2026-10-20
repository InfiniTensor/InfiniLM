#ifndef INFINI_OPS_COMMON_TRAITS_H_
#define INFINI_OPS_COMMON_TRAITS_H_

#include <tuple>
#include <type_traits>

namespace infini::ops {

// --------------------- List and TypePack ---------------------
// A generic container for a sequence of compile-time values.
template <auto... items>
struct List {};

// `ListGet<index>(List<items...>{})` extracts the `i`th value from a `List`
// tag.
template <std::size_t index, auto head, auto... tail>
constexpr auto ListGetImpl(List<head, tail...>) {
  if constexpr (index == 0)
    return head;
  else
    return ListGetImpl<index - 1>(List<tail...>{});
}

template <std::size_t index, auto... items>
constexpr auto ListGet(List<items...> list) {
  return ListGetImpl<index>(list);
}

template <typename... Ts>
struct TypePack {};

// -----------------------------------------------------------------------------
// Tags
// -----------------------------------------------------------------------------
// Tags are passed as regular function arguments to user functors instead of
// template parameters. This lets users write plain C++17 `[](auto tag)` lambdas
// rather than C++20 template lambdas (`[]<typename T>()`).

// `TypeTag<T>`: carries a C++ type. Recover with `typename
// decltype(tag)::type`.
template <typename T>
struct TypeTag {
  using type = T;
};

// `ValueTag<V>`: carries a compile-time value. Recover with
// `decltype(tag)::value`.
template <auto v>
struct ValueTag {
  using value_type = decltype(v);
  static constexpr auto value = v;
};

// -----------------------------------------------------------------------------
// List Queries
// -----------------------------------------------------------------------------

// Check at compile-time if a value exists within a construct (e.g., `List<>`).
// Example: `static_assert(ContainsValue<SupportedTiles, 32>)`;
template <typename T, auto value>
struct Contains;

template <auto value, auto... items>
struct Contains<List<items...>, value>
    : std::disjunction<std::bool_constant<value == items>...> {};

template <typename T, auto value>
inline constexpr bool ContainsValue = Contains<T, value>::value;

// Check at compile-time if a type `T` is present in a variadic list of types
// `Ts`.
// Example: `static_assert(IsTypeInList<T, float, int>)`;
template <typename T, typename... Ts>
inline constexpr bool IsTypeInList = (std::is_same_v<T, Ts> || ...);

// Trait to detect whether `T` is a `List<...>` specialization.
template <typename T>
struct IsListType : std::false_type {};

template <auto... items>
struct IsListType<List<items...>> : std::true_type {};

// -----------------------------------------------------------------------------
// List Operations
// -----------------------------------------------------------------------------

// Concatenates two List types into a single `List`.
// Example: `ConcatType<List<1, 2>, List<3, 4>>` is `List<1, 2, 3, 4>`.
template <typename L1, typename L2>
struct Concat;

template <auto... item1, auto... item2>
struct Concat<List<item1...>, List<item2...>> {
  using type = List<item1..., item2...>;
};

template <typename L1, typename L2>
using ConcatType = typename Concat<L1, L2>::type;

template <typename... Lists>
struct Flatten;

template <auto... items>
struct Flatten<List<items...>> {
  using type = List<items...>;
};

template <typename L1, typename L2, typename... Rest>
struct Flatten<L1, L2, Rest...> {
  using type = typename Flatten<ConcatType<L1, L2>, Rest...>::type;
};

// -----------------------------------------------------------------------------
// Invocability Detection (SFINAE)
// -----------------------------------------------------------------------------

// Checks if a `Functor` can be called with a `ValueTag<Value>` and `Args...`.
template <typename Functor, auto value, typename = void, typename... Args>
struct IsInvocable : std::false_type {};

template <typename Functor, auto value, typename... Args>
struct IsInvocable<Functor, value,
                   std::void_t<decltype(std::declval<Functor>()(
                       ValueTag<value>{}, std::declval<Args>()...))>,
                   Args...> : std::true_type {};

template <typename Functor, auto value, typename... Args>
inline constexpr bool IsInvocableValue =
    IsInvocable<Functor, value, void, Args...>::value;

// -----------------------------------------------------------------------------
// Filtering Logic
// -----------------------------------------------------------------------------

// Recursive template to filter values based on `Functor` support at
// compile-time.
template <typename Functor, typename ArgsTuple, typename Result,
          auto... remaining>
struct Filter;

// Base case: All values processed.
template <typename Functor, typename... Args, auto... filtered>
struct Filter<Functor, std::tuple<Args...>, List<filtered...>> {
  using type = List<filtered...>;
};

// Recursive step: Test the `head` value and accumulate if supported.
template <typename Functor, typename... Args, auto... filtered, auto head,
          auto... tail>
struct Filter<Functor, std::tuple<Args...>, List<filtered...>, head, tail...> {
  using type = typename std::conditional_t<
      IsInvocableValue<Functor, head, Args...> &&
          !ContainsValue<List<filtered...>, head>,
      Filter<Functor, std::tuple<Args...>, List<filtered..., head>, tail...>,
      Filter<Functor, std::tuple<Args...>, List<filtered...>, tail...>>::type;
};

// Interface to filter a `List` type directly.
template <typename Functor, typename ArgsTuple, typename ListType>
struct FilterList;

template <typename Functor, typename... Args, auto... items>
struct FilterList<Functor, std::tuple<Args...>, List<items...>> {
  using type =
      typename Filter<Functor, std::tuple<Args...>, List<>, items...>::type;
};

}  // namespace infini::ops

#endif
