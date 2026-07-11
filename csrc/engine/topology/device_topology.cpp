#include "device_topology.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace infinilm::engine::topology {
namespace {

std::string trim(const std::string &value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool env_truthy(const char *name) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return false;
    }
    const std::string value = to_lower(trim(raw));
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::optional<std::string> getenv_nonempty(const char *name) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    std::string value = trim(raw);
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
}

std::vector<std::string> split_mapping_entries(const std::string &value) {
    std::vector<std::string> entries;
    std::string current;
    for (char ch : value) {
        if (ch == ',' || ch == ';') {
            current = trim(current);
            if (!current.empty()) {
                entries.push_back(current);
            }
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    current = trim(current);
    if (!current.empty()) {
        entries.push_back(current);
    }
    return entries;
}

std::optional<std::string> lookup_device_mapping(const char *env_name, const infinicore::Device &device) {
    auto env_value = getenv_nonempty(env_name);
    if (!env_value.has_value()) {
        return std::nullopt;
    }

    const std::string type_name = to_lower(infinicore::Device::toString(device.getType()));
    const std::string index = std::to_string(device.getIndex());
    const std::vector<std::string> expected_keys = {
        type_name + ":" + index,
        type_name + "." + index,
        type_name + index,
        index,
        "*",
    };

    for (const auto &entry : split_mapping_entries(*env_value)) {
        const auto pos = entry.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string key = to_lower(trim(entry.substr(0, pos)));
        const std::string value = trim(entry.substr(pos + 1));
        if (value.empty()) {
            continue;
        }
        if (std::find(expected_keys.begin(), expected_keys.end(), key) != expected_keys.end()) {
            return value;
        }
    }
    return std::nullopt;
}

std::optional<int> parse_int(const std::string &value) {
    try {
        std::size_t consumed = 0;
        int parsed = std::stoi(value, &consumed, 10);
        if (consumed == value.size()) {
            return parsed;
        }
    } catch (...) {
    }
    return std::nullopt;
}

#ifdef __linux__

struct PciBusIdQuery {
    std::string bus_id;
    std::string provider;
};

std::optional<std::string> read_first_line(const std::string &path) {
    std::ifstream file(path);
    if (!file.good()) {
        return std::nullopt;
    }
    std::string line;
    std::getline(file, line);
    line = trim(line);
    if (line.empty()) {
        return std::nullopt;
    }
    return line;
}

std::vector<std::string> pci_bus_id_candidates(std::string bus_id) {
    bus_id = to_lower(trim(bus_id));
    std::vector<std::string> candidates;
    if (bus_id.empty()) {
        return candidates;
    }

    candidates.push_back(bus_id);
    const auto colon = bus_id.find(':');
    if (colon == std::string::npos) {
        candidates.push_back("0000:" + bus_id);
    } else if (colon == 8) {
        candidates.push_back(bus_id.substr(4));
    } else if (colon != 4) {
        candidates.push_back("0000:" + bus_id);
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

std::optional<PciBusIdQuery> query_nvidia_pci_bus_id(const infinicore::Device &device) {
    const char *library_names[] = {"libcudart.so", "libcudart.so.12", "libcudart.so.11.0"};
    void *handle = nullptr;
    for (const char *name : library_names) {
        handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
        if (handle != nullptr) {
            break;
        }
    }
    if (handle == nullptr) {
        return std::nullopt;
    }

    using GetPciBusIdFn = int (*)(char *, int, int);
    auto get_pci_bus_id = reinterpret_cast<GetPciBusIdFn>(dlsym(handle, "cudaDeviceGetPCIBusId"));
    if (get_pci_bus_id == nullptr) {
        dlclose(handle);
        return std::nullopt;
    }

    char bus_id[64] = {};
    int status = get_pci_bus_id(bus_id, static_cast<int>(sizeof(bus_id)), static_cast<int>(device.getIndex()));
    dlclose(handle);
    if (status != 0 || bus_id[0] == '\0') {
        return std::nullopt;
    }
    return PciBusIdQuery{bus_id, "cuda-runtime"};
}

std::optional<PciBusIdQuery> query_device_pci_bus_id(const infinicore::Device &device) {
    if (auto override_value = lookup_device_mapping("INFINILM_DEVICE_PCI_BUS_IDS", device)) {
        return PciBusIdQuery{*override_value, "env-pci"};
    }

    switch (device.getType()) {
    case infinicore::Device::Type::NVIDIA:
        return query_nvidia_pci_bus_id(device);
    case infinicore::Device::Type::HYGON:
    case infinicore::Device::Type::ILUVATAR:
    case infinicore::Device::Type::METAX:
    case infinicore::Device::Type::MOORE:
    case infinicore::Device::Type::CAMBRICON:
    case infinicore::Device::Type::ASCEND:
    case infinicore::Device::Type::KUNLUN:
    case infinicore::Device::Type::QY:
    case infinicore::Device::Type::ALI:
    case infinicore::Device::Type::CPU:
    case infinicore::Device::Type::COUNT:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<int> query_numa_node_from_pci_bus_id(const std::string &bus_id) {
    for (const auto &candidate : pci_bus_id_candidates(bus_id)) {
        auto numa_node = read_first_line("/sys/bus/pci/devices/" + candidate + "/numa_node");
        if (!numa_node.has_value()) {
            continue;
        }
        auto parsed = parse_int(*numa_node);
        if (parsed.has_value()) {
            return parsed;
        }
    }
    return std::nullopt;
}

std::optional<std::string> read_numa_cpu_list(int numa_node) {
    if (numa_node < 0) {
        return std::nullopt;
    }
    return read_first_line("/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist");
}

bool add_cpu_range(cpu_set_t &set, int begin, int end) {
    if (begin < 0 || end < begin) {
        return false;
    }
    for (int cpu = begin; cpu <= end; ++cpu) {
        if (cpu >= CPU_SETSIZE) {
            return false;
        }
        CPU_SET(cpu, &set);
    }
    return true;
}

bool parse_cpu_list(const std::string &cpu_list, cpu_set_t &set) {
    CPU_ZERO(&set);
    bool any = false;
    std::stringstream ss(cpu_list);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            auto cpu = parse_int(token);
            if (!cpu.has_value() || !add_cpu_range(set, *cpu, *cpu)) {
                return false;
            }
        } else {
            auto begin = parse_int(trim(token.substr(0, dash)));
            auto end = parse_int(trim(token.substr(dash + 1)));
            if (!begin.has_value() || !end.has_value() || !add_cpu_range(set, *begin, *end)) {
                return false;
            }
        }
        any = true;
    }
    return any;
}

std::string format_cpu_set(const cpu_set_t &set) {
    std::ostringstream out;
    bool first = true;
    int cpu = 0;
    while (cpu < CPU_SETSIZE) {
        if (!CPU_ISSET(cpu, &set)) {
            ++cpu;
            continue;
        }
        const int begin = cpu;
        while (cpu + 1 < CPU_SETSIZE && CPU_ISSET(cpu + 1, &set)) {
            ++cpu;
        }
        const int end = cpu;
        if (!first) {
            out << ",";
        }
        out << begin;
        if (end != begin) {
            out << "-" << end;
        }
        first = false;
        ++cpu;
    }
    return out.str();
}

bool intersect_with_current_affinity(cpu_set_t &target) {
    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(0, sizeof(current), &current) != 0) {
        return false;
    }

    bool any = false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        const bool keep = CPU_ISSET(cpu, &target) && CPU_ISSET(cpu, &current);
        if (!keep) {
            CPU_CLR(cpu, &target);
        } else {
            any = true;
        }
    }
    return any;
}

CpuAffinityBinding apply_cpu_list(std::string cpu_list, int numa_node, std::string provider, std::string pci_bus_id) {
    CpuAffinityBinding result;
    result.attempted = true;
    result.numa_node = numa_node;
    result.provider = std::move(provider);
    result.pci_bus_id = std::move(pci_bus_id);

    cpu_set_t target;
    if (!parse_cpu_list(cpu_list, target)) {
        result.reason = "invalid CPU list: " + cpu_list;
        return result;
    }
    if (!intersect_with_current_affinity(target)) {
        result.reason = "target CPU list is outside the process affinity mask: " + cpu_list;
        return result;
    }
    if (sched_setaffinity(0, sizeof(target), &target) != 0) {
        result.reason = std::string("sched_setaffinity failed: ") + std::strerror(errno);
        return result;
    }

    result.applied = true;
    result.cpu_list = format_cpu_set(target);
    return result;
}

CpuAffinityBinding bind_current_thread_to_device_numa_linux(const infinicore::Device &device) {
    CpuAffinityBinding result;
    if (env_truthy("INFINILM_DISABLE_CPU_AFFINITY")) {
        result.reason = "disabled by INFINILM_DISABLE_CPU_AFFINITY";
        return result;
    }

    if (auto cpu_list = getenv_nonempty("INFINILM_CPU_AFFINITY_CPUS")) {
        return apply_cpu_list(*cpu_list, -1, "env-cpu-list", "");
    }

    if (auto numa_override = lookup_device_mapping("INFINILM_DEVICE_NUMA_NODES", device)) {
        auto parsed_node = parse_int(*numa_override);
        if (!parsed_node.has_value()) {
            result.attempted = true;
            result.provider = "env-numa";
            result.reason = "invalid NUMA node: " + *numa_override;
            return result;
        }
        auto cpu_list = read_numa_cpu_list(*parsed_node);
        if (!cpu_list.has_value()) {
            result.attempted = true;
            result.provider = "env-numa";
            result.numa_node = *parsed_node;
            result.reason = "cannot read CPU list for NUMA node " + std::to_string(*parsed_node);
            return result;
        }
        return apply_cpu_list(*cpu_list, *parsed_node, "env-numa", "");
    }

    auto pci_bus_id = query_device_pci_bus_id(device);
    if (!pci_bus_id.has_value()) {
        result.attempted = true;
        result.provider = "topology-provider";
        result.reason = "no PCI bus id provider for " + device.toString();
        return result;
    }

    auto numa_node = query_numa_node_from_pci_bus_id(pci_bus_id->bus_id);
    if (!numa_node.has_value() || *numa_node < 0) {
        result.attempted = true;
        result.provider = pci_bus_id->provider;
        result.pci_bus_id = pci_bus_id->bus_id;
        result.reason = "cannot resolve NUMA node for PCI bus id " + pci_bus_id->bus_id;
        return result;
    }

    auto cpu_list = read_numa_cpu_list(*numa_node);
    if (!cpu_list.has_value()) {
        result.attempted = true;
        result.provider = pci_bus_id->provider;
        result.pci_bus_id = pci_bus_id->bus_id;
        result.numa_node = *numa_node;
        result.reason = "cannot read CPU list for NUMA node " + std::to_string(*numa_node);
        return result;
    }

    return apply_cpu_list(*cpu_list, *numa_node, pci_bus_id->provider, pci_bus_id->bus_id);
}

#else

CpuAffinityBinding bind_current_thread_to_device_numa_linux(const infinicore::Device &) {
    CpuAffinityBinding result;
    result.reason = "CPU affinity binding is only supported on Linux";
    return result;
}

#endif

} // namespace

CpuAffinityBinding bind_current_thread_to_device_numa(const infinicore::Device &device) {
    return bind_current_thread_to_device_numa_linux(device);
}

} // namespace infinilm::engine::topology
