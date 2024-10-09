#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Function for multiplying two 64 bit integers into a 128 bit integer represented by two uint64_t
__host__ __device__ void multiply_uint64_t(const uint64_t& x, const uint64_t& y, uint64_t *high, uint64_t *low) {
    /*
    Parameters:
        x, y -  64 bit numbers to multiply
        high, low - the high and low part of the result
    */
    uint64_t x_low = (x & 0xffffffff); // Get the low 32 bits of x
    uint64_t x_high = (x >> 32); // Get the top 32 bits of x
    uint64_t y_low = (y & 0xffffffff); // Get the low 32 bits of y
    uint64_t y_high = (y >> 32); // Get the top 32 bits of y

    // Calculate x_low * y_low
    uint64_t a = x_low * y_low;
    uint64_t low_32 = (a & 0xffffffff); // the bottommost 32 bits of the result. These are the final 1-32 bits

    // Calcuate x_low * y_high
    uint64_t b = x_low * y_high + (a >> 32); // multiplied by 2^32

    // Calculate x_high * y_low
    uint64_t c = x_high * y_low + (b & 0xffffffff); // multiplied by 2^32

    // Calculate x_high * y_high and return the final result
    *low = (c << 32) + low_32; // The low part
    *high = x_high * y_high + (b >> 32) + (c >> 32); // The high part
}