#ifndef __INFINIOPTEST_OPS_HPP__
#define __INFINIOPTEST_OPS_HPP__
#include "test.hpp"

/*
 * Declare all the tests here
 */
DECLARE_INFINIOP_TEST(gemm)
DECLARE_INFINIOP_TEST(random_sample)
DECLARE_INFINIOP_TEST(mul)

#define REGISTER_INFINIOP_TEST(name)                      \
    {                                                     \
        #name,                                            \
        {                                                 \
            infiniop_test::name::Test::build,             \
            infiniop_test::name::Test::attribute_names(), \
            infiniop_test::name::Test::tensor_names(),    \
        }},

/*
 * Register all the tests here
 */
#define TEST_BUILDER_MAPPINGS                 \
    {                                         \
        REGISTER_INFINIOP_TEST(gemm)          \
        REGISTER_INFINIOP_TEST(random_sample) \
        REGISTER_INFINIOP_TEST(mul)           \
    }

namespace infiniop_test {

// Global variable for {op_name: builder} mappings
extern std::unordered_map<std::string, const TestBuilder> TEST_BUILDERS;

} // namespace infiniop_test

#endif
