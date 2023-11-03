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
//  create pseudo bulk data 
// 

class ASAPpbResult {
    public:
    ASAPpbResult(
    const Eigen::MatrixXf _pb,
    const Eigen::MatrixXf _log_pb,
    const Eigen::MatrixXf _pb_batch,
    const Eigen::MatrixXf _batch_effect,
    const Eigen::MatrixXf _log_batch_effect
    ):
    pb(_pb),
    logpb(_log_pb),
    pb_batch(_pb_batch),
    batch_effect(_batch_effect),
    log_batch_effect(_log_batch_effect)
    {}

    Eigen::MatrixXf pb;
    Eigen::MatrixXf logpb;
    Eigen::MatrixXf pb_batch;
    Eigen::MatrixXf batch_effect;
    Eigen::MatrixXf log_batch_effect;
};

class ASAPpb {
    public:
    ASAPpb(const Eigen::MatrixXf in_ysum,
    const Eigen::MatrixXf in_zsum,
    const Eigen::MatrixXf in_deltasum,
    const Eigen::MatrixXf in_n,
    const Eigen::MatrixXf in_p,
    const RowVec in_size
    ):
    ysum_ds(in_ysum),
    zsum_ds(in_zsum),
    deltasum_db(in_deltasum),
    n_bs(in_n),
    p_bs(in_p),
    size_s(in_size)
    {}

    ASAPpbResult generate_pb();
    
    protected:
        Eigen::MatrixXf ysum_ds;
        Eigen::MatrixXf zsum_ds;
        Eigen::MatrixXf deltasum_db;
        Eigen::MatrixXf n_bs;
        Eigen::MatrixXf p_bs;
        RowVec size_s;

};

//
// ASAP NMF
//
struct ASAPNMFResult {

    ASAPNMFResult(Mat beta_dk_a, Mat beta_dk_b, 
        Mat beta_dk_mean, Mat beta_dk_log_mean,
        Mat theta_nk_mean,Mat theta_nk_log_mean,
        std::vector<Scalar>  in_llik_trace)
        : beta_a{beta_dk_a},beta_b{beta_dk_b}, 
          beta{beta_dk_mean},beta_log{beta_dk_log_mean}, 
          theta{theta_nk_mean},theta_log{theta_nk_log_mean}, 
          llik_trace{in_llik_trace} {}

    Mat beta_a, beta_b;
    Mat beta, beta_log;
    Mat theta, theta_log;
    std::vector<Scalar> llik_trace;

};

class ASAPNMF {
    public:
    ASAPNMF(const Eigen::MatrixXf in_Y,int in_maxK):Y(in_Y),maxK(in_maxK){}

    ASAPNMFResult nmf();    
    protected:
        int maxK;
        Eigen::MatrixXf Y;
};

// 
//  Degree correction NMF model 
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
    ASAPdcNMF(const Eigen::MatrixXf in_Y,int in_maxK, int in_seed):Y(in_Y),maxK(in_maxK),seed(in_seed){}

    ASAPdcNMFResult nmf();
    ASAPdcNMFResult online_nmf(const Eigen::MatrixXf in_beta_a, const Eigen::MatrixXf in_beta_b);
    
    protected:
        int maxK;
        int seed;
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

