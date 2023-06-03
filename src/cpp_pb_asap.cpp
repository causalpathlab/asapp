#include "../include/cpp_asap.hh"

ASAPpbResult ASAPpb::create_pb()
{   

    const std::size_t num_factors=10;
    // const Rcpp::Nullable<Rcpp::NumericMatrix> r_covar = R_NilValue;
    // const Rcpp::Nullable<Rcpp::StringVector> r_batch = R_NilValue;
    const std::size_t rseed = 42;
    const bool verbose = false;
    const std::size_t NUM_THREADS = 1;
    const std::size_t BLOCK_SIZE = 100;
    const bool do_normalize = false;
    const bool do_log1p = false;
    const bool do_row_std = false;
    const std::size_t KNN_CELL = 10;
    const std::size_t BATCH_ADJ_ITER = 100;

    Eigen::MatrixXf pb;
    
    return pb;

}

