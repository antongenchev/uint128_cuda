#ifndef UINT128
#define UINT128

#include <cstdint>
#include <cuda_runtime.h>
#include <uint128_cuda_utils.h>

struct uint128_t {
    uint64_t high;
    uint64_t low;

    // Constructors
    __host__ __device__ uint128_t() : high(0), low(0) {}
    __host__ __device__ uint128_t(uint64_t high, uint64_t low) : high(high), low(low) {}

    // Operator overloading: +
    __host__ __device__ uint128_t operator+(const uint128_t& other) const {
        uint128_t result;
        result.low = low + other.low;
        result.high = high + other.high + (result.low < other.low ? 1 : 0);
        return result;
    }

    // Operator overloading: -
    __host__ __device__ uint128_t operator-(const uint128_t& other) const {
        uint128_t result;
        result.low = low - other.low;
        result.high = high - other.high - (low < other.low ? 1 : 0);
        return result;
    }

    // Operator overloading: *
    __host__  uint128_t operator*(const uint128_t& other) const {
        // (a*2^64 + b)(c*2^64 + d) % 2^128 = (a*d + b*c)*2^64 + b*d % 2^128
        uint128_t result;
        multiply_uint64_t(low, other.low, &result.high, &result.low);
        result.high += low * other.high + high * other.low;
        return result;
    }
};

#endif // UINT128