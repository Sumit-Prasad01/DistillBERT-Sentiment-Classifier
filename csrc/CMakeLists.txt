cmake_minimum_required(VERSION 3.18)
project(sentiment_cpp_accel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional LibTorch Standalone Runner
find_package(Torch QUIET)
if (Torch_FOUND)
    message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
    add_executable(inference_engine inference_engine.cpp)
    target_link_libraries(inference_engine "${TORCH_LIBRARIES}")
    target_compile_definitions(inference_engine PRIVATE USE_TORCH)
    set_property(TARGET inference_engine PROPERTY CXX_STANDARD 17)
else()
    message(STATUS "Torch C++ package not found. Building standalone stub.")
    add_executable(inference_engine inference_engine.cpp)
endif()
