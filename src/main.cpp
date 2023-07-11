#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../include/cpp_asap.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace std;

PYBIND11_MODULE(asapc, m) {
    m.doc() = "CPP ASAP module";;


    py::class_<ASAPpb>(m, "ASAPpb")
    .def(py::init< Eigen::MatrixXf&, Eigen::MatrixXf&, Eigen::MatrixXf&, Eigen::MatrixXf&, Eigen::MatrixXf&, RowVec& >(),
            py::arg("in_ysum"),
            py::arg("in_zsum"),
            py::arg("in_deltasum"),
            py::arg("in_n"),
            py::arg("in_p"),
            py::arg("in_size"))
    .def("generate_pb", &ASAPpb::generate_pb,py::return_value_policy::reference_internal);

    
    py::class_<ASAPpbResult>(m, "ASAPpbResult")
    .def(py::init< Mat, Mat, Mat, Mat, Mat>())
    .def_readwrite("pb", &ASAPpbResult::pb)
    .def_readwrite("logpb", &ASAPpbResult::logpb)
    .def_readwrite("pb_batch", &ASAPpbResult::pb_batch)
    .def_readwrite("batch_effect", &ASAPpbResult::batch_effect)
    .def_readwrite("log_batch_effect", &ASAPpbResult::log_batch_effect);

    py::class_<ASAPdcNMFResult>(m, "ASAPdcNMFResult")
    .def(py::init< Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat, std::vector<Scalar> >())
    .def_readwrite("beta_a", &ASAPdcNMFResult::beta_a)
    .def_readwrite("beta_b", &ASAPdcNMFResult::beta_b)
    .def_readwrite("beta", &ASAPdcNMFResult::beta)
    .def_readwrite("beta_log", &ASAPdcNMFResult::beta_log)
    .def_readwrite("theta", &ASAPdcNMFResult::theta)
    .def_readwrite("theta_log", &ASAPdcNMFResult::theta_log)
    .def_readwrite("col_deg", &ASAPdcNMFResult::col_deg)
    .def_readwrite("row_deg", &ASAPdcNMFResult::row_deg)
    .def_readwrite("llik_trace", &ASAPdcNMFResult::llik_trace);

    py::class_<ASAPdcNMF>(m, "ASAPdcNMF")
    .def(py::init< Eigen::MatrixXf&, int>(),
            py::arg("in_Y"),
            py::arg("in_maxK"))
    .def("nmf", &ASAPdcNMF::nmf,py::return_value_policy::reference_internal)
    .def("online_nmf", &ASAPdcNMF::online_nmf,
            py::arg("in_beta_a"),
            py::arg("in_beta_b"),    
            py::return_value_policy::reference_internal);

    py::class_<ASAPdcNMFPredict>(m, "ASAPdcNMFPredict")
    .def(py::init< Eigen::MatrixXf&, Eigen::MatrixXf&, Eigen::MatrixXf&>(),
            py::arg("in_X"),
            py::arg("in_beta_a"),
            py::arg("in_beta_b"))
    .def("predict", &ASAPdcNMFPredict::predict,py::return_value_policy::reference_internal);

    py::class_<ASAPaltNMFResult>(m, "ASAPaltNMFResult")
    .def(py::init< Mat, Mat, Mat, Mat, Mat, Mat, std::vector<Scalar> >())
    .def_readwrite("beta", &ASAPaltNMFResult::beta)
    .def_readwrite("beta_log", &ASAPaltNMFResult::beta_log)
    .def_readwrite("theta", &ASAPaltNMFResult::theta)
    .def_readwrite("theta_log", &ASAPaltNMFResult::theta_log)
    .def_readwrite("philog", &ASAPaltNMFResult::philog)
    .def_readwrite("rholog", &ASAPaltNMFResult::rholog)
    .def_readwrite("llik_trace", &ASAPaltNMFResult::llik_trace);

    py::class_<ASAPaltNMF>(m, "ASAPaltNMF")
    .def(py::init< Eigen::MatrixXf&, int>(),
            py::arg("in_Y_dn"),
            py::arg("in_maxK"))
    .def("nmf", &ASAPaltNMF::nmf,py::return_value_policy::reference_internal);

    py::class_<ASAPaltNMFPredictResult>(m, "ASAPaltNMFPredictResult")
    .def(py::init< Mat, Mat, Mat, Mat, Mat, Mat>())
    .def_readwrite("beta", &ASAPaltNMFPredictResult::beta)
    .def_readwrite("theta", &ASAPaltNMFPredictResult::theta)
    .def_readwrite("corr", &ASAPaltNMFPredictResult::corr)
    .def_readwrite("latent", &ASAPaltNMFPredictResult::latent)
    .def_readwrite("loglatent", &ASAPaltNMFPredictResult::loglatent)
    .def_readwrite("logtheta", &ASAPaltNMFPredictResult::logtheta);

    py::class_<ASAPaltNMFPredict>(m, "ASAPaltNMFPredict")
    .def(py::init< Eigen::MatrixXf&, Eigen::MatrixXf& >(),
            py::arg("in_Y_dn"),
            py::arg("in_log_x"))
    .def("predict", &ASAPaltNMFPredict::predict,py::return_value_policy::reference_internal);

#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}



