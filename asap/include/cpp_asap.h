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
#include "mmutil.h"
#include "math.h"
#include "gamma_parameter.h"
#include "poisson_nmf_model.h"
#include "latent_matrix.h"


namespace py = pybind11;

struct ASAPResults {
    using IntegerMatrix = typename Eigen::
        Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    ASAPResults(Eigen::MatrixXf in_A,IntegerMatrix in_B,int in_C)
        : A{in_A},B{in_B}, C{in_C} {}

    Eigen::MatrixXf A;
    IntegerMatrix B;
    int C;

    static void defPybind(py::module &m) {
        py::class_<ASAPResults>(m, "ASAPResults")
        .def(py::init< Eigen::MatrixXf,IntegerMatrix, int>())
        .def_readwrite("A", &ASAPResults::A)
        .def_readwrite("B", &ASAPResults::B)
        .def_readwrite("C", &ASAPResults::C);
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

