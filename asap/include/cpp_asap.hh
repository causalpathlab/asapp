#pragma once 

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


struct ASAPNMFResult {

    ASAPNMFResult(Mat in_A,Mat in_B,std::vector<Scalar>  in_llik_trace)
        : A{in_A},B{in_B}, llik_trace{in_llik_trace} {}

    Mat A;
    Mat B;
    std::vector<Scalar> llik_trace;

};

class ASAPNMF {
    public:
    ASAPNMF (const Eigen::MatrixXf in_Y,int in_maxK):Y(in_Y),maxK(in_maxK){}

    ASAPNMFResult nmf();

    protected:
        int maxK;
        Eigen::MatrixXf Y;
};

struct ASAPREGResult {

    ASAPREGResult(Mat in_A)
        : A{in_A}{}

    Mat A;

};

class ASAPREG {
    public:
    ASAPREG (const Eigen::MatrixXf in_Y, const Eigen::MatrixXf in_log_x):Y(in_Y),log_x(in_log_x){}

    ASAPREGResult regression();

    protected:
        Eigen::MatrixXf Y;
        Eigen::MatrixXf log_x;
};

