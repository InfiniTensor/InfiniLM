#ifndef INFINI_RT_DISPATCHER_H_
#define INFINI_RT_DISPATCHER_H_

#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "common/traits.h"
#include "data_type.h"
#include "device.h"

namespace infini::rt {

// -----------------------------------------------------------------------------
// Core Generic Runtime Dispatchers
// -----------------------------------------------------------------------------

namespace detail {

// Implements the dispatch body over a resolved `List<head, tail...>`.
template <typename ValueType, typename Functor, typename... Args, auto head,
          auto... tail>
auto DispatchFuncImpl(ValueType value, Functor&& func,
                      std::string_view context_str, List<head, tail...>,
                      Args&&... args) {
  using ReturnType = decltype(std::forward<Functor>(func)(
      ValueTag<static_cast<ValueType>(head)>{}, std::forward<Args>(args)...));

  // Path for void functions.
  if constexpr (std::is_void_v<ReturnType>) {
    bool handled = ((value == static_cast<ValueType>(tail)
                         ? (std::forward<Functor>(func)(
                                ValueTag<tail>{}, std::forward<Args>(args)...),
                            true)
                         : false) ||
                    ... ||
                    (value == static_cast<ValueType>(head)
                         ? (std::forward<Functor>(func)(
                                ValueTag<head>{}, std::forward<Args>(args)...),
                            true)
                         : false));

    if (!handled) {
      // TODO(lzm): change to logging.
      std::cerr << "dispatch error (void): value " << static_cast<int>(value)
                << " not supported in the context: " << context_str << "\n";
      std::abort();
    }
  }
  // Path for non-void functions.
  else {
    std::optional<ReturnType> result;
    bool handled = ((value == static_cast<ValueType>(tail)
                         ? (result.emplace(std::forward<Functor>(func)(
                                ValueTag<tail>{}, std::forward<Args>(args)...)),
                            true)
                         : false) ||
                    ... ||
                    (value == static_cast<ValueType>(head)
                         ? (result.emplace(std::forward<Functor>(func)(
                                ValueTag<head>{}, std::forward<Args>(args)...)),
                            true)
                         : false));

    if (handled) {
      return *result;
    }
    // TODO(lzm): change to logging.
    std::cerr << "dispatch error (non-void): value " << static_cast<int>(value)
              << " not supported in the context: " << context_str << "\n";
    std::abort();
    return ReturnType{};
  }
}

// Deduces `head`/`tail` from a `List` type via partial specialization,
// then forwards to `DispatchFuncImpl`.
template <typename ValueType, typename Functor, typename FilteredList,
          typename ArgsTuple>
struct DispatchFuncUnwrap;

template <typename ValueType, typename Functor, auto head, auto... tail,
          typename... Args>
struct DispatchFuncUnwrap<ValueType, Functor, List<head, tail...>,
                          std::tuple<Args...>> {
  static auto Call(ValueType value, Functor&& func,
                   std::string_view context_str, Args&&... args) {
    return DispatchFuncImpl(value, std::forward<Functor>(func), context_str,
                            List<head, tail...>{}, std::forward<Args>(args)...);
  }
};

// Empty-list specialization
template <typename ValueType, typename Functor, typename... Args>
struct DispatchFuncUnwrap<ValueType, Functor, List<>, std::tuple<Args...>> {
  static auto Call(ValueType value, Functor&&, std::string_view context_str,
                   Args&&...) {
    // TODO(lzm): change to logging.
    std::cerr << "dispatch error: no allowed values registered for value "
              << static_cast<int64_t>(value)
              << " in the context: " << context_str << "\n";
    std::abort();
  }
};

}  // namespace detail

// (Single Dispatch) Dispatches a runtime value to a compile-time functor.
template <typename ValueType, ValueType... all_values, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  using FilteredPack = typename Filter<Functor, std::tuple<Args...>, List<>,
                                       all_values...>::type;

  return detail::DispatchFuncUnwrap<
      ValueType, Functor, FilteredPack,
      std::tuple<Args...>>::Call(value, std::forward<Functor>(func),
                                 context_str, std::forward<Args>(args)...);
}

// (Multi-Dispatch) Dispatches a vector of runtime values to a compile-time
// functor.
// Base Case: All Dimensions Resolved
template <typename Functor, typename... Args, auto... items>
auto DispatchFunc(const std::vector<int64_t>& values, size_t /*index*/,
                  Functor&& func, std::string_view /*context_str*/,
                  List<items...>, Args&&... args) {
  return std::forward<Functor>(func)(List<items...>{},
                                     std::forward<Args>(args)...);
}

// Forward declaration of the recursive multi-dispatch overload.
template <typename FirstList, typename... RestLists, typename Functor,
          typename... Args, auto... items>
auto DispatchFunc(const std::vector<int64_t>& values, size_t index,
                  Functor&& func, std::string_view context_str, List<items...>,
                  Args&&... args);

// Adapter used in the recursive multi-dispatch case: given a resolved value
// `val` recurse into the next dimension.
template <typename RestListsPack, typename Functor, auto... items>
struct MultiDispatchRecurseAdapter;

template <typename... RestLists, typename Functor, auto... items>
struct MultiDispatchRecurseAdapter<TypePack<RestLists...>, Functor, items...> {
  const std::vector<int64_t>& values;
  size_t next_index;
  Functor& func;
  std::string_view context_str;

  template <auto val, typename... Args>
  auto operator()(ValueTag<val>, Args&&... args) const {
    return DispatchFunc<RestLists...>(values, next_index, func, context_str,
                                      List<items..., val>{},
                                      std::forward<Args>(args)...);
  }
};

template <typename RestListsPack, typename Functor, typename... Args,
          auto... items, auto... allowed>
auto MultiDispatchFirstDim(const std::vector<int64_t>& values, size_t index,
                           Functor& func, std::string_view context_str,
                           List<items...>, List<allowed...>, Args&&... args) {
  static_assert(sizeof...(allowed) > 0,
                "`DispatchFunc` dimension list is empty");
  using EnumType = std::common_type_t<decltype(allowed)...>;

  MultiDispatchRecurseAdapter<RestListsPack, Functor, items...> adapter{
      values, index + 1, func, context_str};

  return DispatchFunc<EnumType, allowed...>(
      static_cast<EnumType>(values.at(index)), adapter, context_str,
      std::forward<Args>(args)...);
}

// (Multi-Dispatch) Recursive Case
template <typename FirstList, typename... RestLists, typename Functor,
          typename... Args, auto... items>
auto DispatchFunc(const std::vector<int64_t>& values, size_t index,
                  Functor&& func, std::string_view context_str, List<items...>,
                  Args&&... args) {
  return MultiDispatchFirstDim<TypePack<RestLists...>>(
      values, index, func, context_str, List<items...>{}, FirstList{},
      std::forward<Args>(args)...);
}

// -----------------------------------------------------------------------------
// High-Level Specialized Dispatchers
// -----------------------------------------------------------------------------
// These provide cleaner and more convenient APIs for common InfiniOps types.

namespace detail {

// Bridges the generic value dispatch layer to the `DataType`-specific type
// dispatch layer.
template <Device::Type kDev, typename Functor>
struct DataTypeAdapter {
  Functor& func;

  template <auto dtype, typename... Args>
  auto operator()(ValueTag<dtype>, Args&&... args) const {
    using T = TypeMapType<kDev, static_cast<DataType>(dtype)>;
    return func(TypeTag<T>{}, std::forward<Args>(args)...);
  }
};

template <Device::Type kDev, typename Functor>
struct DataTypeMultiAdapter {
  Functor& func;

  template <auto... dtypes, typename... Args>
  auto operator()(List<dtypes...>, Args&&... args) const {
    return func(TypeTag<TypeMapType<kDev, static_cast<DataType>(dtypes)>>{}...,
                std::forward<Args>(args)...);
  }
};

template <typename Functor>
struct DeviceAdapter {
  Functor& func;

  template <auto dev, typename... Args>
  auto operator()(ValueTag<dev>, Args&&... args) const {
    return func(ValueTag<dev>{}, std::forward<Args>(args)...);
  }
};

template <typename Functor>
struct DeviceMultiAdapter {
  Functor& func;

  template <auto... devs, typename... Args>
  auto operator()(List<devs...>, Args&&... args) const {
    return func(ValueTag<devs>{}..., std::forward<Args>(args)...);
  }
};

}  // namespace detail

// `DataType` Dispatch
template <Device::Type kDev, DataType... allowed_dtypes, typename Functor,
          typename... Args>
auto DispatchFunc(DataType dtype, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  detail::DataTypeAdapter<kDev, std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<DataType, allowed_dtypes...>(dtype, adapter, context_str,
                                                   std::forward<Args>(args)...);
}

// `DataType` Multi-Dispatch
template <Device::Type kDev, typename... Lists, typename Functor,
          typename... Args>
auto DispatchFunc(std::initializer_list<DataType> dtypes, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  std::vector<int64_t> v;
  for (auto d : dtypes) v.push_back(static_cast<int64_t>(d));

  detail::DataTypeMultiAdapter<kDev, std::remove_reference_t<Functor>> adapter{
      func};
  return DispatchFunc<Lists...>(v, 0, adapter, context_str, List<>{},
                                std::forward<Args>(args)...);
}

// `Device` Dispatch
template <auto... allowed_devices, typename Functor, typename... Args>
auto DispatchFunc(Device::Type device, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  detail::DeviceAdapter<std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<Device::Type,
                      static_cast<Device::Type>(allowed_devices)...>(
      device, adapter, context_str, std::forward<Args>(args)...);
}

// `Device` Multi-Dispatch
template <typename... Lists, typename Functor, typename... Args>
auto DispatchFunc(std::initializer_list<Device::Type> devices, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  std::vector<int64_t> v;
  for (auto d : devices) v.push_back(static_cast<int64_t>(d));

  detail::DeviceMultiAdapter<std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<Lists...>(v, 0, adapter, context_str, List<>{},
                                std::forward<Args>(args)...);
}

template <typename ValueType, typename Functor, typename... Args, auto... items>
auto DispatchFuncListAliasImpl(ValueType value, Functor&& func,
                               std::string_view context_str, List<items...>,
                               Args&&... args) {
  return DispatchFunc<static_cast<std::decay_t<ValueType>>(items)...>(
      value, std::forward<Functor>(func), context_str,
      std::forward<Args>(args)...);
}

template <Device::Type kDev, typename ValueType, typename Functor,
          typename... Args, auto... items>
auto DispatchFuncListAliasImpl(ValueType value, Functor&& func,
                               std::string_view context_str, List<items...>,
                               Args&&... args) {
  return DispatchFunc<kDev, static_cast<std::decay_t<ValueType>>(items)...>(
      value, std::forward<Functor>(func), context_str,
      std::forward<Args>(args)...);
}

// Interface for Generic `List` Aliases (for non-DataType dispatch, e.g. Device)
template <typename ListType, typename ValueType, typename Functor,
          typename... Args,
          typename = std::enable_if_t<IsListType<ListType>::value>>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return DispatchFuncListAliasImpl(value, std::forward<Functor>(func),
                                   context_str, ListType{},
                                   std::forward<Args>(args)...);
}

// Interface for Generic `List` Aliases (for DataType dispatch with device type)
template <Device::Type kDev, typename ListType, typename ValueType,
          typename Functor, typename... Args,
          typename = std::enable_if_t<IsListType<ListType>::value>>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return DispatchFuncListAliasImpl<kDev>(value, std::forward<Functor>(func),
                                         context_str, ListType{},
                                         std::forward<Args>(args)...);
}

// Interface for Any `int64_t`-Convertible Types
template <typename... Lists, typename Functor, typename... Args>
auto DispatchFunc(std::initializer_list<int64_t> keys, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  std::vector<int64_t> v_keys(keys);
  return DispatchFunc<Lists...>(v_keys, 0, std::forward<Functor>(func),
                                context_str, List<>{},
                                std::forward<Args>(args)...);
}

}  // namespace infini::rt

#endif
