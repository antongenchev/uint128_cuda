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

    // Operator overloading: / (128bit/64bit)
    __host__ uint128_t operator/(const uint64_t& other) const {
        // Long division
        // (a*2^96 + b*2^64 + c*2^32 + d)/n = (a/n)*2^96 + (((a%n)*2^64 + b)/n)*2^64 + ...
        uint128_t result;
        result.high = ((high >> 32) / other << 32) + //(a/n)*2^96
                      (((high >> 32) % other << 32) + (high & 0xffffffff)) / other; //(((a%n)*2^64 + b)/n)*2^64
        result.low = (((((((high >> 32) % other << 32) + (high & 0xffffffff)) % other) << 32) +
                     (low >> 32)) / other << 32) + // (((((a%n)*2^64 + b)%n)+c)/n)*2^32
                     (((((((high >> 32) % other << 32) + (high & 0xffffffff)) % other << 32) +
                     (low >> 32)) % other << 32) + (low & 0xffffffff)) / other; // ((((((a%n)*2^64 + b)%n)+c)%n) + d)/n
        return result;
    }

};

#endif // UINT128