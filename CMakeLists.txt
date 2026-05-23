cmake_minimum_required(VERSION 3.14)
project(custard-flow)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

# Enable PCH support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Winvalid-pch" COMPILER_SUPPORTS_PCH)

if(COMPILER_SUPPORTS_PCH)
  message(STATUS "Compiler supports PCH")
else()
  message(FATAL_ERROR "Your compiler does not support PCH.")
endif()

find_package(Torch REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Build the CustardFlow library
add_library(custard-flow
    src/CustardFlow.cpp
)
target_include_directories(custard-flow PUBLIC ${CMAKE_SOURCE_DIR}/include)

enable_testing()

add_executable(
  cf_test
  test/cf_test.cc
)
target_link_libraries(
  cf_test
  GTest::gtest_main
  custard-flow
  ${TORCH_LIBRARIES}
)


add_executable(
  bench
  src/matmul_bench.cpp
)
target_link_libraries(
  bench
  custard-flow
  ${TORCH_LIBRARIES}
  m
)


# Enable intrinsics (SSE/AVX) if needed
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(custard-flow PRIVATE -march=native)
elseif(MSVC)
    target_compile_options(custard-flow PRIVATE /arch:AVX)
endif()

include(GoogleTest)

# Precompiled header
target_precompile_headers(cf_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.h)
gtest_discover_tests(cf_test)