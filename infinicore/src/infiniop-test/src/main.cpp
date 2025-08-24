#include "gguf.hpp"
#include "test.hpp"
#include <infinirt.h>
#include <iostream>

struct ParsedArgs {
    std::string file_path;                          // Mandatory argument: test.gguf file path
    infiniDevice_t device_type = INFINI_DEVICE_CPU; // Default to CPU
    int device_id = 0;                              // CUDA device ID (if specified)
    int warmups = 0;                                // Default to 0 if not given
    int iterations = 0;                             // Default to 0 if not given
    double atol = 0.001;                            // Default absolute tolerance
    double rtol = 0.001;                            // Default relative tolerance
};

void printUsage() {
    std::cout << "Usage:" << std::endl
              << std::endl;
    std::cout << "infiniop-test <test.gguf> [--<device>[:id]] [--warmup <warmups>] [--run <iterations>] [--atol <atol>] [--rtol <rtol>]" << std::endl
              << std::endl;
    std::cout << "  <test.gguf>>" << std::endl;
    std::cout << "    Path to the test gguf file" << std::endl
              << std::endl;
    std::cout << "  --<device>[:id]" << std::endl;
    std::cout << "    (Optional) Specify the device type --(cpu|nvidia|cambricon|ascend|metax|moore|iluvatar|kunlun|sugon) and device ID (optional). CPU by default." << std::endl
              << std::endl;
    std::cout << "  --warmup <warmups>" << std::endl;
    std::cout << "    (Optional) Number of warmups to perform before timing. Default to 0." << std::endl
              << std::endl;
    std::cout << "  --run <iterations>" << std::endl;
    std::cout << "    (Optional) Number of iterations to perform for timing. Default to 0." << std::endl
              << std::endl;
    std::cout << "  --atol <absolute_tolerance>" << std::endl;
    std::cout << "    (Optional) Absolute tolerance for correctness check. Default to 0.001" << std::endl
              << std::endl;
    std::cout << "  --rtol <relative_tolerance>" << std::endl;
    std::cout << "    (Optional) Relative tolerance for correctness check. Default to 0.001" << std::endl
              << std::endl;
    exit(-1);
}

#define PARSE_DEVICE(FLAG, DEVICE)                                 \
    else if (arg.find(FLAG) == 0) {                                \
        size_t colon_pos = arg.find(':');                          \
        args.device_type = DEVICE;                                 \
        if (colon_pos != std::string::npos) {                      \
            args.device_id = std::stoi(arg.substr(colon_pos + 1)); \
        } else {                                                   \
            args.device_id = 0;                                    \
        }                                                          \
    }

ParsedArgs parseArgs(int argc, char *argv[]) {
    if (argc < 2) {
        printUsage();
    }

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage();
    }

    ParsedArgs args;
    args.file_path = argv[1]; // First argument is always the test.gguf file

    std::unordered_map<std::string, std::string> options;

    try {
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg.find("--cpu") == 0) {
                args.device_id = 0;
            }
            PARSE_DEVICE("--nvidia", INFINI_DEVICE_NVIDIA)
            PARSE_DEVICE("--cambricon", INFINI_DEVICE_CAMBRICON)
            PARSE_DEVICE("--ascend", INFINI_DEVICE_ASCEND)
            PARSE_DEVICE("--metax", INFINI_DEVICE_METAX)
            PARSE_DEVICE("--moore", INFINI_DEVICE_MOORE)
            PARSE_DEVICE("--iluvatar", INFINI_DEVICE_ILUVATAR)
            PARSE_DEVICE("--kunlun", INFINI_DEVICE_KUNLUN)
            PARSE_DEVICE("--sugon", INFINI_DEVICE_SUGON)
            else if (arg == "--warmup" && i + 1 < argc) {
                args.warmups = std::stoi(argv[++i]);
            }
            else if (arg == "--run" && i + 1 < argc) {
                args.iterations = std::stoi(argv[++i]);
            }
            else if (arg == "--atol" && i + 1 < argc) {
                args.atol = std::stod(argv[++i]);
            }
            else if (arg == "--rtol" && i + 1 < argc) {
                args.rtol = std::stod(argv[++i]);
            }
            else {
                printUsage();
            }
        }
    } catch (const std::exception &) {
        printUsage();
    }

    return args;
}

int main(int argc, char *argv[]) {
    ParsedArgs args = parseArgs(argc, argv);
    int failed = 0;
    try {
        std::cout << args.file_path << std::endl;
        GGUFFileReader reader = GGUFFileReader(args.file_path);
        // std::cout << reader.toString() << std::endl;

        if (infinirtInit() != INFINI_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to initialize InfiniRT" << std::endl;
            return -1;
        }

        auto results = infiniop_test::runAllTests(
            reader,
            (infiniDevice_t)args.device_type, args.device_id,
            args.warmups, args.iterations,
            args.rtol, args.atol);

        std::cout << "=====================================" << std::endl;
        for (auto result : results) {
            if (!result->isPassed()) {
                failed++;
            }
            std::cout << result->toString() << std::endl;
            std::cout << "=====================================" << std::endl;
        }
        if (failed == 0) {
            std::cout << GREEN << "All tests passed" << RESET << std::endl;
        } else {
            std::cout << RED << failed << " of " << results.size() << " tests failed" << RESET << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return failed;
}
