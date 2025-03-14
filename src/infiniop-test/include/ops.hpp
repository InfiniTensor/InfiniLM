#ifndef __INFINIOPTEST_OPS_HPP__
#define __INFINIOPTEST_OPS_HPP__
#include "test.hpp"

/*
 * Declare all the tests here
 */
DECLARE_INFINIOP_TEST(matmul)

#define REGISTER_INFINIOP_TEST(name)                    \
    {                                                   \
        #name,                                          \
        { infiniop_test::name::Test::build,             \
          infiniop_test::name::Test::attribute_names(), \
          infiniop_test::name::Test::tensor_names() }   \
    }

/*
 * Register all the tests here
 */
#define TEST_BUILDER_MAPPINGS           \
    {                                   \
        REGISTER_INFINIOP_TEST(matmul), \
    }

namespace infiniop_test {
extern std::unordered_map<std::string, const TestBuilder> TEST_BUILDERS;

} // namespace infiniop_test

#endif
