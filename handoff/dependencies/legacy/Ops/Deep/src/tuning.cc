#include "tuning.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string_view>
#include <utility>

namespace infini::ops {

namespace {

using Json = nlohmann::json;

constexpr int kTuningCacheVersion = 1;
constexpr char kDefaultTuningPath[] = "tuning.json";

bool EqualsIgnoreCase(std::string_view value, std::string_view expected) {
  return value.size() == expected.size() &&
         std::equal(value.begin(), value.end(), expected.begin(),
                    [](char left, char right) {
                      return std::toupper(static_cast<unsigned char>(left)) ==
                             std::toupper(static_cast<unsigned char>(right));
                    });
}

bool EnvFlag(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !*value) return false;

  const std::string_view flag{value};
  return flag == "1" || EqualsIgnoreCase(flag, "ON") ||
         EqualsIgnoreCase(flag, "TRUE");
}

int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;

  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

const Json* FindMember(const Json& object, const char* name) {
  if (!object.is_object()) return nullptr;

  auto iterator = object.find(name);
  return iterator == object.end() ? nullptr : &*iterator;
}

bool IsInteger(const Json& value) {
  return value.is_number_integer() || value.is_number_unsigned();
}

template <auto... device_types>
std::optional<Device::Type> DeviceTypeFromString(std::string_view name,
                                                 List<device_types...>) {
  const Device::Type types[]{device_types...};
  for (auto type : types) {
    if (name == Device::StringFromType(type)) return type;
  }
  return std::nullopt;
}

Json SignatureToJson(const TuningSignature& signature) {
  Json tensors = Json::array();
  for (const auto& tensor : signature.tensors) {
    tensors.push_back(
        {{"shape", tensor.shape}, {"dtype", static_cast<int>(tensor.dtype)}});
  }

  return {{"tensors", std::move(tensors)}, {"scalars", signature.scalars}};
}

std::optional<TuningSignature> SignatureFromJson(const Json& value) {
  const Json* tensors = FindMember(value, "tensors");
  const Json* scalars = FindMember(value, "scalars");
  if (!tensors || !tensors->is_array() || !scalars || !scalars->is_array()) {
    return std::nullopt;
  }

  TuningSignature parsed;
  for (const auto& tensor : *tensors) {
    const Json* shape = FindMember(tensor, "shape");
    const Json* dtype = FindMember(tensor, "dtype");
    if (!shape || !shape->is_array() || !dtype || !IsInteger(*dtype)) {
      return std::nullopt;
    }

    TuningSignature::TensorSig tensor_signature;
    for (const auto& dimension : *shape) {
      if (!IsInteger(dimension)) return std::nullopt;

      tensor_signature.shape.push_back(dimension.get<int64_t>());
    }
    tensor_signature.dtype = static_cast<DataType>(dtype->get<int64_t>());
    parsed.tensors.push_back(std::move(tensor_signature));
  }

  for (const auto& scalar : *scalars) {
    if (!scalar.is_number()) return std::nullopt;

    parsed.scalars.push_back(scalar.get<double>());
  }

  return parsed;
}

}  // namespace

TuningManager& TuningManager::Instance() {
  static TuningManager instance;
  return instance;
}

void TuningManager::InitializeFromEnvironment() {
  if (!EnvFlag("INFINI_OPS_ENABLE_AUTO_TUNING")) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = false;
    cache_.clear();
    json_path_.clear();
    return;
  }

  warmup_count_ = EnvInt("INFINI_OPS_TUNING_WARMUP", kDefaultWarmupCount);
  repeat_count_ = EnvInt("INFINI_OPS_TUNING_REPEAT", kDefaultRepeatCount);

  const char* path = std::getenv("INFINI_OPS_TUNING_PATH");
  LoadTuningCache(path && *path ? path : kDefaultTuningPath);
}

void TuningManager::LoadTuningCache(const std::string& json_path) {
  std::lock_guard<std::mutex> lock(mutex_);

  cache_.clear();
  json_path_ = json_path;
  enabled_ = true;

  std::ifstream file(json_path);
  if (!file.is_open()) return;

  Json root = Json::parse(file, nullptr, false);
  const Json* version = FindMember(root, "version");
  const Json* entries = FindMember(root, "entries");
  if (root.is_discarded() || !version || !IsInteger(*version) || !entries ||
      !entries->is_array()) {
    std::cerr << "[TuningManager] Warning: failed to parse " << json_path
              << ", starting with an empty cache" << std::endl;
    cache_.clear();
    return;
  }

  if (version->get<int>() != kTuningCacheVersion) {
    std::cerr << "[TuningManager] Warning: tuning.json version "
              << version->get<int>() << " not supported (expected "
              << kTuningCacheVersion << ")" << std::endl;
    return;
  }

  for (const auto& entry : *entries) {
    const Json* operator_name = FindMember(entry, "operator");
    const Json* device_name = FindMember(entry, "device");
    const Json* signature_json = FindMember(entry, "signature");
    const Json* best_implementation = FindMember(entry, "best_implementation");
    if (!operator_name || !operator_name->is_string() || !device_name ||
        !device_name->is_string() || !signature_json || !best_implementation ||
        !IsInteger(*best_implementation)) {
      continue;
    }

    auto device = DeviceTypeFromString(
        device_name->get_ref<const std::string&>(), AllDeviceTypes{});
    auto signature = SignatureFromJson(*signature_json);
    if (!device.has_value() || !signature.has_value()) {
      continue;
    }

    CacheKey key{operator_name->get_ref<const std::string&>(), *device,
                 std::move(*signature)};
    cache_[std::move(key)] = best_implementation->get<std::size_t>();
  }

  std::cout << "[TuningManager] Loaded " << cache_.size()
            << " tuning entries from " << json_path << std::endl;
}

std::optional<std::size_t> TuningManager::Lookup(
    std::string_view operator_name, Device::Type device,
    const TuningSignature& signature) const {
  if (!enabled_) return std::nullopt;

  std::lock_guard<std::mutex> lock(mutex_);
  CacheKey key{std::string{operator_name}, device, signature};
  auto iterator = cache_.find(key);
  if (iterator == cache_.end()) return std::nullopt;

  return iterator->second;
}

void TuningManager::Record(std::string_view operator_name, Device::Type device,
                           const TuningSignature& signature,
                           std::size_t best_index) {
  if (!enabled_) return;

  std::lock_guard<std::mutex> lock(mutex_);
  CacheKey key{std::string{operator_name}, device, signature};
  cache_[key] = best_index;
  FlushToDiskLocked();
}

void TuningManager::FlushToDiskLocked() const {
  std::ofstream out(json_path_, std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "[TuningManager] Warning: cannot write tuning cache to "
              << json_path_ << std::endl;
    return;
  }

  Json entries = Json::array();
  for (const auto& [key, best_implementation] : cache_) {
    entries.push_back(
        {{"operator", key.operator_name},
         {"device", std::string{Device::StringFromType(key.device)}},
         {"signature", SignatureToJson(key.signature)},
         {"best_implementation", best_implementation}});
  }

  Json root{{"version", kTuningCacheVersion}, {"entries", std::move(entries)}};
  out << root.dump(2) << '\n';
}

}  // namespace infini::ops
