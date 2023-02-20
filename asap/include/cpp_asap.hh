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

struct ASAPResults {

    ASAPResults(Mat in_A,Mat in_B,std::vector<Scalar>  in_llik_trace)
        : A{in_A},B{in_B}, llik_trace{in_llik_trace} {}

    Mat A;
    Mat B;
    std::vector<Scalar> llik_trace;

    static void defPybind(py::module &m) {
        py::class_<ASAPResults>(m, "ASAPResults")
        .def(py::init< Mat, Mat, std::vector<Scalar> >())
        .def_readwrite("A", &ASAPResults::A)
        .def_readwrite("B", &ASAPResults::B)
        .def_readwrite("C", &ASAPResults::llik_trace);
    }

};

class ASAP {

    public:

    ASAP (const Eigen::MatrixXf in_Y,int in_maxK):Y(in_Y),maxK(in_maxK){}

    static void defPybind(py::module &m) {
        py::class_<ASAP>(m, "ASAP")
        .def(py::init< Eigen::MatrixXf&, int>(),
                py::arg("in_Y"),
                py::arg("in_maxK"))
        .def("run", &ASAP::nmf,py::return_value_policy::reference_internal);
        }

    ASAPResults nmf();

    protected:

        int maxK;

        Eigen::MatrixXf Y;



};

