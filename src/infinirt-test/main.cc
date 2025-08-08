#include "test.h"
#include <infinirt.h>

struct ParsedArgs {
    infiniDevice_t device_type = INFINI_DEVICE_CPU;
};

void printUsage() {
    std::cout << "Usage:" << std::endl
              << "  infinirt-test [--<device>]" << std::endl
              << std::endl
              << "Options:" << std::endl
              << "  --<device>   Specify the device type." << std::endl
              << std::endl
              << "Available devices:" << std::endl
              << "  cpu         - Default" << std::endl
              << "  nvidia" << std::endl
              << "  cambricon" << std::endl
              << "  ascend" << std::endl
              << "  metax" << std::endl
              << "  moore" << std::endl
              << "  iluvatar" << std::endl
              << "  kunlun" << std::endl
              << "  sugon" << std::endl
              << std::endl;
    exit(EXIT_FAILURE);
}

ParsedArgs parseArgs(int argc, char *argv[]) {
    ParsedArgs args;

    if (argc < 2) {
        return args; // 默认使用 CPU
    }

    std::string arg = argv[1];
    if (arg == "--help" || arg == "-h") {
        printUsage();
    }

    try {
#define PARSE_DEVICE(FLAG, DEVICE) \
    if (arg == FLAG) {             \
        args.device_type = DEVICE; \
    }
        // clang-format off
        PARSE_DEVICE("--cpu", INFINI_DEVICE_CPU)
        else PARSE_DEVICE("--nvidia", INFINI_DEVICE_NVIDIA)
        else PARSE_DEVICE("--cambricon", INFINI_DEVICE_CAMBRICON)
        else PARSE_DEVICE("--ascend", INFINI_DEVICE_ASCEND)
        else PARSE_DEVICE("--metax", INFINI_DEVICE_METAX)
        else PARSE_DEVICE("--moore", INFINI_DEVICE_MOORE)
        else PARSE_DEVICE("--iluvatar", INFINI_DEVICE_ILUVATAR)
        else PARSE_DEVICE("--kunlun", INFINI_DEVICE_KUNLUN)
        else PARSE_DEVICE("--sugon", INFINI_DEVICE_SUGON)
        else {
            printUsage();
        }
        // clang-format on
#undef PARSE_DEVICE
    } catch (const std::exception &) {
        printUsage();
    }

    return args;
}

int main(int argc, char *argv[]) {

    ParsedArgs args = parseArgs(argc, argv);
    std::cout << "Testing Device: " << args.device_type << std::endl;
    infiniDevice_t device = args.device_type;

    // 获取设备总数
    std::vector<int> deviceCounts(INFINI_DEVICE_TYPE_COUNT, 0);
    if (infinirtGetAllDeviceCount(deviceCounts.data()) != INFINI_STATUS_SUCCESS) {
        std::cerr << "Failed to get total device count." << std::endl;
        return 1;
    }

    int numDevices = deviceCounts[device];
    std::cout << "Device Type: " << device << " | Available Devices: " << numDevices << std::endl;

    if (numDevices == 0) {
        std::cout << "Device type " << device << " has no available devices." << std::endl;
        return 0;
    }

    for (int deviceId = 0; deviceId < numDevices; ++deviceId) {
        if (!testSetDevice(device, deviceId)) {
            return 1;
        }

        size_t dataSize[] = {1 << 10, 4 << 10, 2 << 20, 1L << 30};

        for (size_t size : dataSize) {
            if (!testMemcpy(device, deviceId, size)) {
                return 1;
            }
        }
    }

    return 0;
}
