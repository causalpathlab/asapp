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


    Mat delta_db = Mat::Ones(D, B);
    
    Mat mu_ds = Mat::Ones(D, S);
    Mat log_mu_ds = Mat::Ones(D, S);

    Mat delta_denom_db = Mat::Zero(D, B); 

    Mat gamma_ds = Mat::Ones(D, S); 

    for (std::size_t t = 0; t < BATCH_ADJ_ITER; ++t) {

        std::cout<<t<<std::endl;

        // Mat a = delta_db;
        // std::cout<<a.cols()<<std::endl;
        // std::cout<<a.rows()<<std::endl;

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
    std::cout<<"ok"<<std::endl;
    
    return mu_ds;

}

