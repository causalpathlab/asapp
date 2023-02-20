#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../include/cpp_asap.hh"

namespace py = pybind11;

using namespace std;

PYBIND11_MODULE(asapc, m) {
    m.doc() = "CPP ASAP module";;
    ASAPResults::defPybind(m);
    ASAP::defPybind(m);
}