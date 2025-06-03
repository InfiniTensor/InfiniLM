#ifndef __INFINIOPTEST_HPP__
#define __INFINIOPTEST_HPP__

#include "gguf.hpp"
#include "tensor.hpp"
#include <functional>
#include <sstream>
#include <unordered_map>
#include <vector>

#define RESET "\033[0m"
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"

namespace infiniop_test {
enum class TestStatus {
    PASS,
    TEST_INIT_FAILED,
    OP_CREATION_FAILED,
    OP_EXECUTION_FAILED,
    RESULT_INCORRECT,
};

// Result of a testcase
class Result {
private:
    TestStatus _status;
    double _time = 0.;
    std::string _description;
    std::string _error_message;

public:
    Result(TestStatus status_, double time_, const std::string &description_, const std::string &error_message_)
        : _status(status_), _time(time_), _description(description_), _error_message(error_message_) {}
    bool isPassed() const { return _status == TestStatus::PASS; }
    std::string toString() const;
};

// Quick macro for creating a test result
#define TEST_PASSED(delay) std::make_shared<infiniop_test::Result>(infiniop_test::TestStatus::PASS, delay, toString(), "")
#define TEST_FAILED(reason, msg) std::make_shared<infiniop_test::Result>(infiniop_test::TestStatus::reason, 0., toString(), msg)
#define TEST_INIT_FAILED(op_name) std::make_shared<infiniop_test::Result>(infiniop_test::TestStatus::TEST_INIT_FAILED, 0., "Invalid " + std::string(op_name), "")

// Run all tests read from a GGUF file
std::vector<std::shared_ptr<Result>> runAllTests(
    const GGUFFileReader &,
    infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations,
    double rtol, double atol);

// Run a single test read from a GGUF file
std::shared_ptr<Result> runTest(
    const GGUFFileReader &,
    infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations,
    double rtol, double atol,
    size_t test_id);

// Check if two tensors are close within given tolerance
void allClose(std::shared_ptr<Tensor> actual, std::shared_ptr<Tensor> expected, double rtol = 1e-3, double atol = 1e-3);

// Helper function for benchmarking a function
double benchmark(std::function<void()> func, size_t warmups, size_t iterations);
} // namespace infiniop_test

namespace infiniop_test::base {
// Base class for a testcase, each operator test should inherit from this class
class Test {
public:
    virtual std::shared_ptr<infiniop_test::Result> run(
        infiniopHandle_t handle, infiniDevice_t device, int device_id,
        size_t warm_ups, size_t iterations)
        = 0;
    virtual std::string toString() const = 0;
};

} // namespace infiniop_test::base

// Quick macro for declaring a new testcase
#define DECLARE_INFINIOP_TEST(name)                                           \
    namespace infiniop_test::name {                                           \
    class Test : public infiniop_test::base::Test {                           \
        double _rtol, _atol;                                                  \
                                                                              \
    public:                                                                   \
        static std::string op_name() { return #name; }                        \
        static std::shared_ptr<Test> build(                                   \
            std::unordered_map<std::string, std::vector<uint8_t>> attributes, \
            std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors, \
            double, double);                                                  \
                                                                              \
        static std::vector<std::string> attribute_names();                    \
        static std::vector<std::string> tensor_names();                       \
        static std::vector<std::string> output_names();                       \
                                                                              \
        std::shared_ptr<infiniop_test::Result> run(                           \
            infiniopHandle_t handle, infiniDevice_t device, int device_id,    \
            size_t warm_ups, size_t iterations) override;                     \
                                                                              \
        std::string toString() const override;                                \
                                                                              \
        ~Test();                                                              \
                                                                              \
    private:                                                                  \
        struct Attributes;                                                    \
        Attributes *_attributes;                                              \
        Test() = delete;                                                      \
        Test(double rtol, double atol) : _rtol(rtol), _atol(atol) {}          \
    };                                                                        \
    }

namespace infiniop_test {
using BuilderFunc = std::function<std::shared_ptr<infiniop_test::base::Test>(
    std::unordered_map<std::string, std::vector<uint8_t>>,
    std::unordered_map<std::string, std::shared_ptr<Tensor>>,
    double, double)>;

// Testcase Registry
// Each testcase should provid a formatted builder, attribute names, and tensor names
struct TestBuilder {
    BuilderFunc build;
    std::vector<std::string> attribute_names;
    std::vector<std::string> tensor_names;
    std::vector<std::string> output_names;
};
} // namespace infiniop_test

#endif
