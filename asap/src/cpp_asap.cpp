#include "../include/cpp_asap.hh"


ASAPNMFResult ASAPNMF::nmf()
{   

    const std::size_t mcem = 10;
    const std::size_t burnin = 10;
    const std::size_t latent_iter = 10;
    const std::size_t thining = 3;
    const bool verbose = true;
    const bool eval_llik = true;
    const double a0 = 1.;
    const double b0 = 1.;
    const std::size_t seed = 42;
    const std::size_t NUM_THREADS = 1;
    const bool update_loading = true;
    const bool gibbs_sampling = false;

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);


    using RNG = dqrng::xoshiro256plus;
    RNG rng(seed);

    using gamma_t = gamma_param_t<Mat, RNG>;

    poisson_nmf_t<Mat, RNG, gamma_t> model(D, N, K, a0, b0, seed);


    using latent_t = latent_matrix_t<RNG>;
    latent_t aux(D, N, K, rng);

    model.initialize_degree(Y);

    aux.randomize();
    model.initialize_by_svd(Y);


    Scalar llik;
    std::vector<Scalar> llik_trace;

    if (eval_llik) {
        llik = model.log_likelihood(Y, aux);
        llik_trace.emplace_back(llik);
    }

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {


        if (update_loading) {
            model.update_topic_loading(Y, aux);
        }
        model.update_row_topic(Y, aux);
        model.update_column_topic(Y, aux);
        model.update_degree(Y);

        if (eval_llik && t % thining == 0) {
            llik = model.log_likelihood(Y, aux);
            llik_trace.emplace_back(llik);
        }
    }

    ASAPNMFResult result{model.row_topic.mean(),model.column_topic.mean(), llik_trace};

    return result;
}

ASAPREGResult ASAPREG::regression()
{
    const double a0 = 1.;
    const double b0 = 1.;
    const std::size_t max_iter = 10;
    const bool verbose = false;
    const std::size_t NUM_THREADS = 1;
    const std::size_t BLOCK_SIZE = 100;

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = log_x.cols(); // number of topics
    const Index block_size = BLOCK_SIZE;
    const Scalar TOL = 1e-20;

    auto exp_op = [](const Scalar &_x) -> Scalar { return fasterexp(_x); };
    
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Mat log_X = standardize(log_x);
    const RowVec Xsum = log_X.unaryExpr(exp_op).colwise().sum();

    Mat Z_tot(N, K);
    Mat theta_tot(N, K);
    Mat log_theta_tot(N, K);


    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;
    softmax_op_t<Mat> softmax;

    ColVec Ysum = Y.colwise().sum().transpose(); // N x 1
    gamma_t theta_b(Y.cols(), K, a0, b0, rng);   // N x K
    Mat log_Z(Y.cols(), K), Z(Y.cols(), K);      // N x K
    Mat R = (Y.transpose() * log_X).array().colwise() / Ysum.array();
    //          N x D        D x K                      N x 1

    ColVec onesN(N); // N x 1
    onesN.setOnes(); //

    for (std::size_t t = 0; t < max_iter; ++t) {

        log_Z = theta_b.log_mean() + R;
        for (Index i = 0; i < Y.cols(); ++i) {
            Z.row(i) = softmax(log_Z.row(i));
        }

        for (Index k = 0; k < K; ++k) {
            const Scalar xk = Xsum(k);
            theta_b.update_col(Z.col(k).cwiseProduct(Ysum), onesN * xk, k);
        }
        theta_b.calibrate();
    }


    ASAPREGResult regresult{theta_b.mean()};

    return regresult;

};