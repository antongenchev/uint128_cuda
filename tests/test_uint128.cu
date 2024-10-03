#include <gtest/gtest.h>
#include "uint128_cuda.h"

TEST(UInt128Test, Addition) {
    uint128_t a(0, 1);
    uint128_t b(0, 2);
    uint128_t result = a + b;
    EXPECT_EQ(result.low, 3);
    EXPECT_EQ(result.high, 0);
}

TEST(UInt128Test, AdditionOverflow) {
    uint64_t max_uint64 = UINT64_MAX;
    uint128_t max_val(0, max_uint64);
    uint128_t result = max_val + uint128_t(0, 1);
    EXPECT_EQ(result.high, 1);
    EXPECT_EQ(result.low, 0);
}