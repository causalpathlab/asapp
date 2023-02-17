#include "cpp_asap.h"


ASAPResults ASAP::nmf()
{   

    py::print("Starting nmf...");

    const std::size_t mcem = 100;
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

    model.initialize_degree(Y);
    model.initialize_by_svd(Y);

    Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(10, 10);

    ASAPResults result{model.onesN, model.onesD, maxK};

    return result;
}