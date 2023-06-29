#include "../include/cpp_asap.hh"

ASAPpbResult ASAPpb::generate_pb()
{   

    const double a0 = 1.;
    const double b0 = 1.;
    const std::size_t seed = 42;
    const std::size_t BATCH_ADJ_ITER = 100;

    using RNG = dqrng::xoshiro256plus;
    RNG rng(seed);

    const Index D = ysum_ds.rows();
    const Index S = ysum_ds.cols();
    const Index B = deltasum_db.cols();

    
    gamma_param_t<Mat, RNG> delta_param(D, B, a0, b0, rng);
    gamma_param_t<Mat, RNG> mu_param(D, S, a0, b0, rng);
    gamma_param_t<Mat, RNG> gamma_param(D, S, a0, b0, rng);

    // batch specific bias for each gene
    Mat delta_db = Mat::Ones(D, B);
    Mat log_delta_db = Mat::Ones(D, B);
    Mat delta_denom_db = Mat::Zero(D, B);
    Mat delta_ds; 
    
    // batch effect free expression for each pseudobulk sample
    Mat mu_ds = Mat::Ones(D, S);
    Mat log_mu_ds = Mat::Ones(D, S);

    // counterfactual bias
    Mat gamma_ds = Mat::Ones(D, S); 


    for (std::size_t t = 0; t < BATCH_ADJ_ITER; ++t) {

        mu_param.update(ysum_ds + zsum_ds,
                        delta_db * n_bs +
                            ((gamma_ds.array().rowwise() * size_s.array()))
                                .matrix());
        mu_param.calibrate();
        mu_ds = mu_param.mean();


        gamma_param
            .update(zsum_ds,
                    (mu_ds.array().rowwise() * size_s.array()).matrix());
        gamma_param.calibrate();
        gamma_ds = gamma_param.mean();

        delta_denom_db = mu_ds * n_bs.transpose();
        delta_param.update(deltasum_db, delta_denom_db);
        delta_param.calibrate();
        delta_db = delta_param.mean();
    }
    
    log_delta_db = delta_param.log_mean();
    delta_ds = delta_db * p_bs;

    mu_param.update(ysum_ds, delta_db * n_bs);
    mu_param.calibrate();
    mu_ds = mu_param.mean();
    log_mu_ds = mu_param.log_mean();

    ASAPpbResult res{mu_ds,log_mu_ds,delta_ds,delta_db,log_delta_db};
    
    return res;

}

