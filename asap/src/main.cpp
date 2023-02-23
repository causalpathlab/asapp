#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../include/cpp_asap.hh"

namespace py = pybind11;

using namespace std;

PYBIND11_MODULE(asapc, m) {
    m.doc() = "CPP ASAP module";;

    py::class_<ASAPNMFResult>(m, "ASAPNMFResult")
    .def(py::init< Mat, Mat, std::vector<Scalar> >())
    .def_readwrite("A", &ASAPNMFResult::A)
    .def_readwrite("B", &ASAPNMFResult::B)
    .def_readwrite("C", &ASAPNMFResult::llik_trace);

    py::class_<ASAPNMF>(m, "ASAPNMF")
    .def(py::init< Eigen::MatrixXf&, int>(),
            py::arg("in_Y"),
            py::arg("in_maxK"))
    .def("run", &ASAPNMF::nmf,py::return_value_policy::reference_internal);

    py::class_<ASAPREGResult>(m, "ASAPREGResult")
    .def(py::init< Mat>())
    .def_readwrite("A", &ASAPREGResult::A);

    py::class_<ASAPREG>(m, "ASAPREG")
    .def(py::init< Eigen::MatrixXf&, Eigen::MatrixXf& >(),
            py::arg("in_Y"),
            py::arg("in_log_x"))
    .def("regress", &ASAPREG::regression,py::return_value_policy::reference_internal);

}



