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
#include "dcpoisson_nmf_model.hh"
#include "latent_matrix.hh"


struct ASAPNMFAltResult {

    ASAPNMFAltResult(Mat beta_dk_mean, Mat beta_dk_log, 
        Mat theta_dk_mean, Mat theta_dk_log,
        Mat logphi_dk, Mat logrho_nk, 
        std::vector<Scalar>  in_llik_trace)
        : beta{beta_dk_mean},beta_log{beta_dk_log}, 
          theta{theta_dk_mean},theta_log{theta_dk_log}, 
          philog{logphi_dk},rholog{logrho_nk}, 
          llik_trace{in_llik_trace} {}

    Mat beta, beta_log;
    Mat theta, theta_log;
    Mat philog;
    Mat rholog;
    std::vector<Scalar> llik_trace;

};

class ASAPNMFAlt {
    public:
    ASAPNMFAlt(const Eigen::MatrixXf in_Y_dn,int in_maxK):Y_dn(in_Y_dn),maxK(in_maxK){}

    ASAPNMFAltResult nmf();

    protected:
        int maxK;
        Eigen::MatrixXf Y_dn;
};

struct ASAPNMFResult {

    ASAPNMFResult(Mat in_A,Mat in_B,std::vector<Scalar>  in_llik_trace)
        : A{in_A},B{in_B}, llik_trace{in_llik_trace} {}

    Mat A;
    Mat B;
    std::vector<Scalar> llik_trace;

};

struct ASAPNMFDCResult {

    ASAPNMFDCResult(Mat in_A, Mat in_B, Mat in_C, Mat in_D, std::vector<Scalar>  in_llik_trace)
        : A{in_A},B{in_B},C{in_C},D{in_D},llik_trace{in_llik_trace} {}

    Mat A;
    Mat B;
    Mat C;
    Mat D;
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

class ASAPNMFDC {
    public:
    ASAPNMFDC (const Eigen::MatrixXf in_Y,int in_maxK):Y(in_Y),maxK(in_maxK){}

    ASAPNMFDCResult nmf();

    protected:
        int maxK;
        Eigen::MatrixXf Y;
};

struct ASAPREGResult {

    ASAPREGResult(Mat in_beta,Mat in_theta, Mat in_corr, Mat in_latent, Mat in_loglatent, Mat in_logtheta)
        :beta{in_beta}, theta{in_theta}, corr{in_corr}, latent{in_latent}, loglatent{in_loglatent}, logtheta{in_logtheta}{}

    Mat beta;
    Mat theta;
    Mat corr;
    Mat latent;
    Mat loglatent;
    Mat logtheta;
};

class ASAPREG {
    public:
    ASAPREG (const Eigen::MatrixXf in_Y, const Eigen::MatrixXf in_log_x):Y_dn(in_Y),log_x(in_log_x){}

    ASAPREGResult regression();

    protected:
        Eigen::MatrixXf Y_dn;
        Eigen::MatrixXf log_x;
};

