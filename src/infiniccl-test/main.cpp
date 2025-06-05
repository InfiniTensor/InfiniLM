#include "infiniccl_test.hpp"

#include <iostream>

struct ParsedArgs {
    infiniDevice_t device_type;
};

void printUsage() {
    std::cout << "Usage:" << std::endl
              << std::endl;
    std::cout << "infiniccl-test --<device>" << std::endl
              << std::endl;
    std::cout << "  --<device>" << std::endl;
    std::cout << "    Specify the device type --(nvidia|cambricon|ascend|metax|moore|iluvatar|kunlun|sugon)." << std::endl
              << std::endl;
    std::cout << "The program will run tests on all visible devices of the specified device type."
              << " Use Environmental Variables such as CUDA_VSIBLE_DEVICES to limit visible device IDs.";
    exit(-1);
}

#define PARSE_DEVICE(FLAG, DEVICE) \
    if (arg == FLAG) {             \
        args.device_type = DEVICE; \
    }

ParsedArgs parseArgs(int argc, char *argv[]) {
    if (argc != 2) {
        printUsage();
    }

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage();
    }

    ParsedArgs args;
    try {
        std::string arg = argv[1];
        // clang-format off
        PARSE_DEVICE("--nvidia", INFINI_DEVICE_NVIDIA)
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

    } catch (const std::exception &) {
        printUsage();
    }

    return args;
}

int main(int argc, char *argv[]) {
    ParsedArgs args = parseArgs(argc, argv);
    int ndevice = 0;
    infinirtInit();
    if (infinirtGetDeviceCount(args.device_type, &ndevice) != INFINI_STATUS_SUCCESS) {
        std::cout << "Failed to get device count" << std::endl;
        return -1;
    }
    if (ndevice == 0) {
        std::cout << "No devices found. Tests skipped." << std::endl;
        return 0;
    } else {
        std::cout << "Found " << ndevice << " devices. Running tests..." << std::endl;
    }

    int failed = 0;
    failed += testAllReduce(args.device_type, ndevice);
    return failed;
}
