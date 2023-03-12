#include <iostream>
#include "../include/cpp_asap.hh"


    
int main()
{
Eigen::MatrixXf mat = Eigen::MatrixXf::Random(4, 3);

int maxk = 2;

ASAPdcNMF dcmodel(mat,maxk);
dcmodel.nmf();

ASAPaltNMF altmodel(mat,maxk);
altmodel.nmf();

// Eigen::MatrixXf y = Eigen::MatrixXf::Random(4, 3);
// Eigen::MatrixXf beta = Eigen::MatrixXf::Random(4, 2);

// ASAPREG model(y,beta);
// model.regression();

    // const std::size_t seed = 42;

    // using RNG = dqrng::xoshiro256plus;
    // RNG rng(seed);

    // const Scalar a0(100.);
    // const Scalar b0(.01);

    // Mat a_stat(10,5);
    // Mat b_stat(10,5);

    // a_stat.setConstant(a0);
    // b_stat.setConstant(b0);

    //     struct rgamma_op_t {

    // explicit rgamma_op_t(RNG &_rng)
    //     : rng(_rng)
    // {
    // }

    // using gamma_distrib = boost::random::gamma_distribution<Scalar>;

    // const Scalar operator()(const Scalar &a, const Scalar &b) const
    // {
    //     return _rgamma(rng, typename gamma_distrib::param_type(a, b));
    // }

    // RNG &rng;
    // gamma_distrib _rgamma;
    // };

    // rgamma_op_t rgamma_op(rng);
    
    // Mat z = a_stat.binaryExpr(b_stat, rgamma_op);

    // z = z * 100;


std::cout<<"ok"<<std::endl;
return 0;
}
