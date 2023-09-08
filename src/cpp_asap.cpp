#include "../include/cpp_asap.hh"

ASAPNMFResult ASAPNMF::nmf()
{   

    // std::cout << "Running nmf...." << '\n';
    
    const std::size_t mcem = 100;
    const std::size_t burnin = 10;
    const std::size_t thining = 3;
    const bool verbose = true;
    const bool eval_llik = true;
    const double a0 = 1.;
    const double b0 = 1.;
    const std::size_t seed = 42;

    const Index D = Y.rows();
    const Index N = Y.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);


    using RNG = dqrng::xoshiro256plus;
    RNG rng(seed);

    using gamma_t = gamma_param_t<Mat, RNG>;

    poisson_nmf_t<Mat, RNG, gamma_t> model(D, N, K, a0, b0, seed);

    model.initialize();

    Scalar llik;
    std::vector<Scalar> llik_trace;
    
    if (eval_llik) {
        llik = model.log_likelihood(Y);
        llik_trace.emplace_back(llik);
    }

    for (std::size_t t = 0; t < (mcem + burnin); ++t) {

    model.update_xaux(Y);
    model.update_column_topic(Y);

    model.update_xaux(Y);
    model.update_row_topic(Y);

        if (eval_llik && t % thining == 0) {
            llik = model.log_likelihood(Y);
            llik_trace.emplace_back(llik);
        }

    }

    ASAPNMFResult result{model.row_topic.a_stat, model.row_topic.b_stat,
        model.row_topic.mean(),model.row_topic.log_mean(),
        model.column_topic.mean(),model.column_topic.log_mean(),
        llik_trace};

    return result;
}
