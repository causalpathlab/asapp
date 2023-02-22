#pragma once 

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>


#include "xoshiro.h"

#include "math.hh"
#include "mmutil.hh"
#include "gamma_parameter.hh"
#include "poisson_nmf_model.hh"
#include "latent_matrix.hh"


namespace py = pybind11;

struct ASAPNMFResult {

    ASAPNMFResult(Mat in_A,Mat in_B,std::vector<Scalar>  in_llik_trace)
        : A{in_A},B{in_B}, llik_trace{in_llik_trace} {}

    Mat A;
    Mat B;
    std::vector<Scalar> llik_trace;

    static void defPybind(py::module &m) {
        py::class_<ASAPNMFResult>(m, "ASAPNMFResult")
        .def(py::init< Mat, Mat, std::vector<Scalar> >())
        .def_readwrite("A", &ASAPNMFResult::A)
        .def_readwrite("B", &ASAPNMFResult::B)
        .def_readwrite("C", &ASAPNMFResult::llik_trace);
    }

};

class ASAPNMF {
    public:
    ASAPNMF (const Eigen::MatrixXf in_Y,int in_maxK):Y(in_Y),maxK(in_maxK){}
    static void defPybind(py::module &m) {
        py::class_<ASAPNMF>(m, "ASAPNMF")
        .def(py::init< Eigen::MatrixXf&, int>(),
                py::arg("in_Y"),
                py::arg("in_maxK"))
        .def("run", &ASAPNMF::nmf,py::return_value_policy::reference_internal);
        }

    ASAPNMFResult nmf();

    protected:
        int maxK;
        Eigen::MatrixXf Y;
};

struct ASAPREGResult {

    ASAPREGResult(Mat in_A)
        : A{in_A}{}

    Mat A;
    static void defPybind(py::module &m) {
        py::class_<ASAPREGResult>(m, "ASAPREGResult")
        .def(py::init< Mat>())
        .def_readwrite("A", &ASAPREGResult::A);
    }

};

class ASAPREG {
    public:
    ASAPREG (const Eigen::MatrixXf in_Y, const Eigen::MatrixXf in_log_x):Y(in_Y),log_x(in_log_x){}
    static void defPybind(py::module &m) {
        py::class_<ASAPREG>(m, "ASAPREG")
        .def(py::init< Eigen::MatrixXf&, Eigen::MatrixXf& >(),
                py::arg("in_Y"),
                py::arg("in_log_x"))
        .def("regress", &ASAPREG::regression,py::return_value_policy::reference_internal);
        }

    ASAPREGResult regression();

    protected:
        Eigen::MatrixXf Y;
        Eigen::MatrixXf log_x;
};

