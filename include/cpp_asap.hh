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

// 
//  Degree correction PNMF model 
// 

struct ASAPdcNMFResult {

    ASAPdcNMFResult(Mat beta_dk_a, Mat beta_dk_b, 
        Mat beta_dk_mean, Mat beta_dk_log_mean,
        Mat theta_nk_mean,Mat theta_nk_log_mean,
        Mat col_deg_mean, 
        Mat row_deg_mean, 
        std::vector<Scalar>  in_llik_trace)
        : beta_a{beta_dk_a},beta_b{beta_dk_b}, 
          beta{beta_dk_mean},beta_log{beta_dk_log_mean}, 
          theta{theta_nk_mean},theta_log{theta_nk_log_mean}, 
          col_deg{col_deg_mean},row_deg{row_deg_mean}, 
          llik_trace{in_llik_trace} {}

    Mat beta_a, beta_b;
    Mat beta, beta_log;
    Mat theta, theta_log;
    Mat col_deg;
    Mat row_deg;
    std::vector<Scalar> llik_trace;

};

class ASAPdcNMF {
    public:
    ASAPdcNMF(const Eigen::MatrixXf in_Y,int in_maxK):Y(in_Y),maxK(in_maxK){}

    ASAPdcNMFResult nmf();
    
    protected:
        int maxK;
        Eigen::MatrixXf Y;
};

class ASAPdcNMFPredict {
    public:
    ASAPdcNMFPredict(const Eigen::MatrixXf in_X, const Eigen::MatrixXf in_beta_a, const Eigen::MatrixXf in_beta_b):X(in_X), beta_a(in_beta_a),beta_b(in_beta_b){}

    ASAPdcNMFResult predict();

    protected:
        Eigen::MatrixXf X;
        Eigen::MatrixXf beta_a;
        Eigen::MatrixXf beta_b;
};


// 
//  Alternating Regression PNMF model 
// 

struct ASAPaltNMFResult {

    ASAPaltNMFResult(Mat beta_dk_mean, Mat beta_dk_log, 
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

class ASAPaltNMF {
    public:
    ASAPaltNMF(const Eigen::MatrixXf in_Y_dn,int in_maxK):Y_dn(in_Y_dn),maxK(in_maxK){}

    ASAPaltNMFResult nmf();

    protected:
        int maxK;
        Eigen::MatrixXf Y_dn;
};


struct ASAPaltNMFPredictResult {

    ASAPaltNMFPredictResult(Mat in_beta,Mat in_theta, Mat in_corr, Mat in_latent, Mat in_loglatent, Mat in_logtheta)
        :beta{in_beta}, theta{in_theta}, corr{in_corr}, latent{in_latent}, loglatent{in_loglatent}, logtheta{in_logtheta}{}

    Mat beta;
    Mat theta;
    Mat corr;
    Mat latent;
    Mat loglatent;
    Mat logtheta;
};

class ASAPaltNMFPredict {
    public:
    ASAPaltNMFPredict (const Eigen::MatrixXf in_Y, const Eigen::MatrixXf in_log_x):Y_dn(in_Y),log_x(in_log_x){}

    ASAPaltNMFPredictResult predict();

    protected:
        Eigen::MatrixXf Y_dn;
        Eigen::MatrixXf log_x;
};

