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

    // Operator overloading: * (128bit*64bit)
    __host__ uint128_t operator*(const uint64_t& other) const {
        uint128_t result;
        multiply_uint64_t(low, other, &result.high, &result.low);
        result.high += high * other;
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

    // Operator overloading: <
    __host__ bool operator<(const uint128_t& other) const {
        if (high != other.high) {
            return high < other.high; // Compare the high parts first
        }
        return low < other.low; // If high parts are equal, compare the low parts
    }

    // Operator overloading: >
    bool operator>(const uint128_t& other) const {
        return other < *this; // a > b if b < a
    }

    // Operator overloading: <=
    bool operator<=(const uint128_t& other) const {
        return !(*this > other); // a <= b if it's not the case that a > b
    }

    // Operator overloading: <=
    bool operator>=(const uint128_t& other) const {
        return !(*this < other); // a >= b if it's not the case that a < b
    }

    // Operator overloading: ==
    bool operator==(const uint128_t& other) const {
        return (low == other.low) and (high == other.high);
    }

    // Operator overloading: / (128bit/128bit)
    __host__ uint128_t operator/(const uint128_t& other) const {
        if (other.high == 0) {
            // If the divisor can fit in 64 bits then yse (128bit/64bit) division
            return *this / other.low;
        } else if (*this <= other) {
            // If the denominator is not smaller than the numerator then the result is trivial
            return (this->low == other.low ? uint128_t(0,1) : uint128_t(0,0));
        }
        // Approximate the answer. (A_1A_0 / D_1__) is an upper boundary for (A1_0A_0 / D_1D_0)
        // Also (A1A0 / D_1__)*0.5 is a lower boundary for (A1_0A_0 / D_1D_0)
        uint64_t upper_bound(this->high / other.high);
        uint64_t lower_bound = upper_bound / 2;
        // Binary search refiment to find the exact result
        uint64_t result = lower_bound + (upper_bound - lower_bound) / 2; // upper_bound + lower_bound can overflow
        uint128_t result_times_divisor = other * result;
        uint128_t this_minus_other = *this - other;
        while (true) {
            if (result_times_divisor > *this ) {
                // result is too big we should decrease it
                upper_bound = result - 1;
                // return result_times_divisor;
            } else if (result_times_divisor <= *this - other) {
                // result is too small we should increase it
                lower_bound = result + 1;
            } else {
                // the result is just right
                break;
            }
            result = lower_bound + (upper_bound - lower_bound) / 2;
            result_times_divisor = other * result;
        }
        return uint128_t(0, result);
    }

};

#endif // UINT128