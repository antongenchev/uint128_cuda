#ifndef UINT128
#define UINT128

#include <cstdint>
#include <cuda_runtime.h>

struct uint128_t {
    uint64_t high;
    uint64_t low;

    __host__ __device__ uint128_t() : high(0), low(0) {}
    __host__ __device__ uint128_t(uint64_t high, uint64_t low) : high(high), low(low) {}
};


#endif // UINT128