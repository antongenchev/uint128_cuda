cmake_minimum_required(VERSION 3.16)
project(uint128_cuda LANGUAGES CUDA CXX)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Add the source files
add_executable(uint128_cuda_test tests/test_uint128.cu)

# Link Google Test to the test executable
target_link_libraries(uint128_cuda_test gtest_main)

# Specify CUDA runtime and include directory
target_include_directories(uint128_cuda_test PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Ensure that the target uses CUDA
set_target_properties(uint128_cuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD_REQUIRED ON
)

# Add the tests to CTest
add_test(NAME uint128_cuda_test COMMAND uint128_cuda_test)