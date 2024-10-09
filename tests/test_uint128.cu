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

TEST(UInt128Test, Subtraction) {
    uint128_t a(2,5);
    uint128_t b(2,3);
    uint128_t result = a - b;
    EXPECT_EQ(result.high, 0);
    EXPECT_EQ(result.low, 2);
}

TEST(UInt128Test, SubtractionOverflow) {
    uint128_t a(1,0);
    uint128_t b(0,UINT64_MAX);
    uint128_t result = a - b;
    EXPECT_EQ(result.high, 0);
    EXPECT_EQ(result.low, 1);
}

TEST(Uint128Test, Multiplication1) {
    uint128_t a(0, 1);
    uint128_t b(0, 2);
    uint128_t result = a * b;
    EXPECT_EQ(result.high, 0);
    EXPECT_EQ(result.low, 2);
}

TEST(Uint128Test, Multiplication2) {
    uint128_t a(0, UINT64_MAX);
    uint128_t b(0, 2);
    uint128_t result = a * b;
    EXPECT_EQ(result.high, 1);
    EXPECT_EQ(result.low, UINT64_MAX - 1);
    uint128_t result2 = b * a;
    EXPECT_EQ(result.low, result2.low);
    EXPECT_EQ(result.high, result2.high);
}

TEST(Uint128Test, Multiplication3) {
    uint128_t a(0, UINT64_MAX);
    uint128_t b(0, UINT64_MAX);
    uint128_t result = a * b;
    EXPECT_EQ(result.high, UINT64_MAX - 1);
    EXPECT_EQ(result.low, 1);
    uint128_t result2 = b * a;
    EXPECT_EQ(result.low, result2.low);
    EXPECT_EQ(result.high, result2.high);
}

TEST(Uint128Test, Multiplication4) {
    uint128_t a(UINT64_MAX, UINT64_MAX);
    uint128_t b(UINT64_MAX, UINT64_MAX);
    uint128_t result = a * b;
    EXPECT_EQ(result.high, 0);
    EXPECT_EQ(result.low, 1);
    uint128_t result2 = b * a;
    EXPECT_EQ(result.low, result2.low);
    EXPECT_EQ(result.high, result2.high);
}

TEST(Uint128Test, DivisionBy64bit1) {
    uint128_t a(0, 6);
    uint64_t b(2);
    uint128_t result = a / b;
    EXPECT_EQ(result.high, 0);
    EXPECT_EQ(result.low, 3);
}

TEST(Uint128Test, DivisionBy64bit2) {
    uint128_t a(0, INT64_MAX);
    uint64_t b = 2;
    uint128_t result = a / b;
    EXPECT_EQ(result.high, 0);
    EXPECT_EQ(result.low, (1ULL << 62) - 1);
}

TEST(Uint128Test, DivisionBy64bit3) {
    uint128_t a(3, 0);
    uint64_t b = 2;
    uint128_t result = a / b;
    EXPECT_EQ(result.high, 1);
    EXPECT_EQ(result.low, (1ULL << 63));
}

TEST(Uint128Test, DivisionBy64bit4) {
    uint128_t a(INT64_MAX, INT64_MAX - 1);
    uint64_t b = 2;
    uint128_t result = a / b;
    EXPECT_EQ(result.high, INT64_MAX >> 1);
    // EXPECT_EQ(result.low, INT64_MAX);
}

TEST(Uint128Test, Comparison1) {
    uint128_t a(1, 1);
    uint128_t b(1, 1);
    EXPECT_EQ(a == b, true);
    EXPECT_EQ(a >= b, true);
    EXPECT_EQ(a <= b, true);
    EXPECT_EQ(a > b, false);
    EXPECT_EQ(a < b, false);
}

TEST(Uint128Test, Comparison2) {
    uint128_t a(1, 1);
    uint128_t b(1, 2);
    EXPECT_EQ(a == b, false);
    EXPECT_EQ(a >= b, false);
    EXPECT_EQ(a <= b, true);
    EXPECT_EQ(a > b, false);
    EXPECT_EQ(a < b, true);
}

TEST(Uint128Test, Comparison3) {
    uint128_t a(1, 2);
    uint128_t b(2, 1);
    EXPECT_EQ(a == b, false);
    EXPECT_EQ(a >= b, false);
    EXPECT_EQ(a <= b, true);
    EXPECT_EQ(a > b, false);
    EXPECT_EQ(a < b, true);
}
