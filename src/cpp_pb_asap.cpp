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


    Mat delta_db, delta_sd_db, log_delta_db, log_delta_sd_db, delta_ds;
    Mat prob_bs, n_bs;


    delta_db.resize(D, B); 
    delta_db.setOnes();

    Mat delta_denom_db = Mat::Zero(D, B); 

    Mat gamma_ds = Mat::Ones(D, S); 



    Eigen::MatrixXf pb;

    pb.setZero();
    
    return pb;

}

