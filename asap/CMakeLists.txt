cmake_minimum_required(VERSION 3.1)
project(asapc)

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "include/*.h")

add_subdirectory("external/pybind11")
pybind11_add_module(asapc
    ${HEADER_FILES}
    ${SOURCE_FILES}
    )