#pragma once

#include <infinicore/device.hpp>

#include <string>

namespace infinilm::engine::topology {

struct CpuAffinityBinding {
    bool attempted = false;
    bool applied = false;
    int numa_node = -1;
    std::string cpu_list;
    std::string pci_bus_id;
    std::string provider;
    std::string reason;
};

CpuAffinityBinding bind_current_thread_to_device_numa(const infinicore::Device &device);

} // namespace infinilm::engine::topology
