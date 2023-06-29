#include <iostream>
#include "../include/cpp_asap.hh"


    
int main()
{

// Eigen::MatrixXf mat = Eigen::MatrixXf::Random(4, 3);

// int maxk = 2;

// ASAPdcNMF dcmodel(mat,maxk);
// dcmodel.nmf();

// ASAPaltNMF altmodel(mat,maxk);
// altmodel.nmf();

Eigen::MatrixXf ysum_ds = Eigen::MatrixXf::Random(100, 25);
Eigen::MatrixXf zsum_ds = Eigen::MatrixXf::Random(100, 25);
Eigen::MatrixXf deltasum_db = Eigen::MatrixXf::Random(100, 2);
Eigen::MatrixXf n_bs = Eigen::MatrixXf::Random(2,25);
Eigen::MatrixXf p_bs = Eigen::MatrixXf::Random(2,25);
RowVec size_s = RowVec::Ones(25);


ASAPpb model(ysum_ds,zsum_ds,deltasum_db,n_bs,p_bs,size_s);
model.generate_pb();

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
