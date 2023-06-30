#include "../include/cpp_asap.hh"

ASAPaltNMFResult ASAPaltNMF::nmf()
{
    const std::size_t max_iter = 100;
    const std::size_t burnin = 10;
    const bool verbose = true;
    const double a0 = 1.;
    const double b0 = 1.;
    const std::size_t rseed = 42;
    const double EPS = 1e-4;

    const Index D = Y_dn.rows();
    const Index N = Y_dn.cols();
    const Index K = std::min(static_cast<Index>(maxK), N);

    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;
    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

    using RNG = dqrng::xoshiro256plus;
    RNG rng(rseed);
    using gamma_t = gamma_param_t<Mat, RNG>;

    gamma_t beta_dk(D, K, a0, b0, rng);  // dictionary
    gamma_t theta_nk(N, K, a0, b0, rng); // scaling for all the factor loading

    Mat logPhi_dk(D, K), phi_dk(D, K); // row to topic latent assignment
    Mat logRho_nk(N, K), rho_nk(N, K); // column to topic latent assignment

    using norm_dist_t = boost::random::normal_distribution<Scalar>;
    norm_dist_t norm_dist(0., 1.);
    auto rnorm = [&rng, &norm_dist]() -> Scalar { return norm_dist(rng); };
    auto exp_op = [](const Scalar x) -> Scalar { return fasterexp(x); };
    auto at_least_one = [](const Scalar x) -> Scalar {
        return (x < 1.) ? 1. : x;
    };

    const ColVec Y_n = Y_dn.colwise().sum().transpose();
    const ColVec Y_d = Y_dn.transpose().colwise().sum();
    const ColVec Y_n1 = Y_n.unaryExpr(at_least_one);
    const ColVec Y_d1 = Y_d.unaryExpr(at_least_one);
    const ColVec ones_n = ColVec::Ones(N);
    const ColVec ones_d = ColVec::Ones(D);

    softmax_op_t<Mat> softmax;

    ////////////////////
    // Initialization //
    ////////////////////

    logPhi_dk = Mat::NullaryExpr(D, K, rnorm);
    for (Index ii = 0; ii < D; ++ii) {
        phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
    }

    logRho_nk = Mat::NullaryExpr(N, K, rnorm);
    for (Index jj = 0; jj < N; ++jj) {
        rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
    }

    Mat X_nk(N, K), X_dk(D, K);

    for (Index tt = 0; tt < burnin; ++tt) {
        X_nk = standardize(logRho_nk, EPS);
        logPhi_dk = Y_dn * X_nk;
        logPhi_dk.array().colwise() /= Y_d1.array();

        X_dk = standardize(logPhi_dk, EPS);
        logRho_nk = Y_dn.transpose() * X_dk;
        logRho_nk.array().colwise() /= Y_n1.array();
        for (Index ii = 0; ii < D; ++ii) {
            phi_dk.row(ii) = softmax.apply_row(logPhi_dk.row(ii));
        }

        for (Index jj = 0; jj < N; ++jj) {
            rho_nk.row(jj) = softmax.apply_row(logRho_nk.row(jj));
        }
    }

    {
        // Column: update theta_k
        theta_nk.update(Y_dn.transpose() * phi_dk,                //
                        ones_n * beta_dk.mean().colwise().sum()); //
        theta_nk.calibrate();

        // Update row topic factors
        beta_dk.update(Y_dn * rho_nk,                             //
                       ones_d * theta_nk.mean().colwise().sum()); //
        beta_dk.calibrate();
    }

    std::vector<Scalar> llik_trace;
    llik_trace.reserve(max_iter);

    RowVec tempK(K);

    for (Index tt = 0; tt < max_iter; ++tt) {

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (i,k)  //
        //////////////////////////////////////////////

        X_nk = standardize(theta_nk.log_mean(), EPS);
        logPhi_dk = Y_dn * X_nk;
        logPhi_dk.array().colwise() /= Y_d1.array();
        logPhi_dk += beta_dk.log_mean();

        for (Index ii = 0; ii < D; ++ii) {
            tempK = logPhi_dk.row(ii);
            logPhi_dk.row(ii) = softmax.log_row(tempK);
        }
        phi_dk = logPhi_dk.unaryExpr(exp_op);

        // Update column topic factors, theta(j, k)
        theta_nk.update(rho_nk.cwiseProduct(Y_dn.transpose() * phi_dk), //
                        ones_n * beta_dk.mean().colwise().sum());       //
        theta_nk.calibrate();

        // Update row topic factors
        beta_dk.update((phi_dk.array().colwise() * Y_d.array()).matrix(), //
                       ones_d * theta_nk.mean().colwise().sum());         //
        beta_dk.calibrate();

        //////////////////////////////////////////////
        // Estimation of auxiliary variables (j,k)  //
        //////////////////////////////////////////////

        X_dk = standardize(beta_dk.log_mean(), EPS);
        logRho_nk = Y_dn.transpose() * X_dk;
        logRho_nk.array().colwise() /= Y_n1.array();
        logRho_nk += theta_nk.log_mean();

        for (Index jj = 0; jj < N; ++jj) {
            tempK = logRho_nk.row(jj);
            logRho_nk.row(jj) = softmax.log_row(tempK);
        }
        rho_nk = logRho_nk.unaryExpr(exp_op);

        // Update row topic factors
        beta_dk.update(phi_dk.cwiseProduct(Y_dn * rho_nk),        //
                       ones_d * theta_nk.mean().colwise().sum()); //
        beta_dk.calibrate();

        // Update column topic factors
        theta_nk.update((rho_nk.array().colwise() * Y_n.array()).matrix(), //
                        ones_n * beta_dk.mean().colwise().sum());          //
        theta_nk.calibrate();

        // evaluate log-likelihood
        Scalar llik = (phi_dk.cwiseProduct(beta_dk.log_mean()).transpose() *
                       Y_dn * rho_nk)
                          .sum();
            
        llik += (rho_nk.cwiseProduct(theta_nk.log_mean()).transpose() * 
                Y_dn.transpose() * phi_dk)
                .sum();
        llik -= (ones_d.transpose() * beta_dk.mean() *
                 theta_nk.mean().transpose() * ones_n)
                    .sum();

        llik_trace.emplace_back(llik);

        const Scalar diff =
            tt > 0 ? abs(llik_trace.at(tt - 1) - llik) / abs(llik + EPS) : 0;

        if (tt > 0 && diff < EPS) {
            break;
        }
    }

    ASAPaltNMFResult res(beta_dk.mean(), beta_dk.log_mean(), theta_nk.mean(),theta_nk.log_mean(),logPhi_dk, logRho_nk, llik_trace);

    return res;
}

ASAPaltNMFPredictResult ASAPaltNMFPredict::predict()
{
    const double a0 = 1.;
    const double b0 = 1.;
    const std::size_t max_iter = 100;
    const bool verbose = false;
    const std::size_t NUM_THREADS = 1;

    using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;
    using ColVec = typename Eigen::internal::plain_col_type<Mat>::type;

    const Index D = Y_dn.rows();
    const Index N = Y_dn.cols();
    const Index K = log_x.cols(); // number of topics
    const Scalar TOL = 1e-20;
    const double EPS = 1e-4;

    auto exp_op = [](const Scalar &_x) -> Scalar { return fasterexp(_x); };
    
    auto at_least_one = [](const Scalar x) -> Scalar {
        return (x < 1.) ? 1. : x;
    };

    // const Mat log_X = standardize(log_x);
    const Mat log_X = log_x;
    const RowVec Xsum = log_X.unaryExpr(exp_op).colwise().sum();

    Mat R_tot(N, K);
    Mat Z_tot(N, K);
    Mat logZ_tot(N, K);
    Mat theta_tot(N, K);
    Mat log_theta_tot(N, K);


    Mat Y = Mat(Y_dn);
    using RNG = dqrng::xoshiro256plus;
    using gamma_t = gamma_param_t<Mat, RNG>;
    RNG rng;
    softmax_op_t<Mat> softmax;

    ColVec Ysum = Y.colwise().sum().transpose(); // n x 1
    ColVec Ysum1 = Ysum.unaryExpr(at_least_one);

    gamma_t theta_b(Y.cols(), K, a0, b0, rng);   // n x K
    Mat logZ(Y.cols(), K), Z(Y.cols(), K);       // n x K
    Mat R = Y.transpose() * log_X;     // n x D        D x K                   
    R.array().colwise() /= Ysum1.array();

    ColVec onesN(Y.cols()); // n x 1
    onesN.setOnes();        //

    for (std::size_t t = 0; t < max_iter; ++t) {

        logZ = theta_b.log_mean() + R;
        for (Index i = 0; i < Y.cols(); ++i) {
            Z.row(i) = softmax.apply_row(logZ.row(i));
        }

        for (Index k = 0; k < K; ++k) {
            const Scalar xk = Xsum(k);
            theta_b.update_col(Z.col(k).cwiseProduct(Ysum), onesN * xk, k);
        }
        theta_b.calibrate();
    }

    ASAPaltNMFPredictResult regresult(log_X.unaryExpr(exp_op),theta_b.mean(), R, Z, logZ, theta_b.log_mean());

    return regresult;

};