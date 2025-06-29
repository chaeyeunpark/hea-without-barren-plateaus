project(RandomHamiltonianGrad LANGUAGES CXX)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 20)

option(ENABLE_PYTHON "Enable python binding" ON)
option(ENABLE_WARNINGS "Enable all warnings" ON)
option(ENABLE_CPP_RUNNER "Enalbe C++ runner" OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_TESTS OFF)

################################################################################
#                              Set dependencies
################################################################################

include(FetchContent)

if(ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()


find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

if(ENABLE_CPP_RUNNER)
    find_package(MPI REQUIRED)
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
endif()

if(ENABLE_WARNINGS)
    add_library(kicked_ising_compile_flags INTERFACE)
    target_compile_options(kicked_ising_compile_flags INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()
 

add_library(kicked_ising_utils STATIC src/utils.cpp)
target_link_libraries(kicked_ising_utils PRIVATE kicked_ising_compile_flags)
target_link_libraries(kicked_ising_utils PUBLIC Eigen3::Eigen)
set_target_properties(kicked_ising_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kicked_ising_utils PUBLIC external)


if(ENABLE_CPP_RUNNER)
    add_executable(runner_mpi src/runner_mpi.cpp)
    target_link_libraries(runner_mpi PRIVATE Eigen3::Eigen BLAS::BLAS kicked_ising_compile_flags kicked_ising_utils MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_compile_options(runner_mpi PRIVATE "-DEIGEN_USE_BLAS")
endif()

if(ENABLE_PYTHON)
    pybind11_add_module(kicked_ising_mbl src/binding.cpp)
    target_link_libraries(kicked_ising_mbl PUBLIC kicked_ising_utils kicked_ising_compile_flags)
    set_target_properties(kicked_ising_mbl PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
