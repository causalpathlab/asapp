#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "cpp_asap.h"

namespace py = pybind11;

using namespace std;

PYBIND11_MODULE(CPP_ASAP, m) {
    m.doc() = "CPP ASAP module";;
    ASAPResults::defPybind(m);
    ASAP::defPybind(m);
}