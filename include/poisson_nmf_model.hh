#include "svd.hh"
#include "eigen_util.hh"
#include "mmutil.hh"

#ifndef POISSON_NMF_MODEL_HH_
#define POISSON_NMF_MODEL_HH_

template <typename T, typename RNG, typename PARAM>
struct poisson_nmf_t {

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;

    explicit poisson_nmf_t(const Index d,
                           const Index n,
                           const Index k,
                           const Scalar a0,
                           const Scalar b0,
                           std::size_t rseed = 42)
        : D(d)
        , N(n)
        , K(k)
        , onesD(d, 1)
        , onesN(n, 1)
        , rng(rseed)
        , row_topic(d, k, a0, b0, rng)
        , column_topic(n, k, a0, b0, rng)
        , x_aux(d,n)
    {
        onesD.setOnes();
        onesN.setOnes();
        x_aux.setOnes();
    }

    void initialize()
    {
        column_topic.initialize();
        row_topic.initialize();
    }

    template <typename Derived>
    const Scalar log_likelihood(const Eigen::MatrixBase<Derived> &Y)
    {

        Mat exp_logc = column_topic.log_mean().unaryExpr(exp_op).transpose();
        Mat exp_logr = row_topic.log_mean().unaryExpr(exp_op);
        Mat exp_log = (exp_logr * exp_logc).unaryExpr(log_op);       

        Mat m1 = Y.array() * exp_log.array();
        Mat m2 =  row_topic.mean() * column_topic.mean().transpose();

        return (m1-m2).sum();

    }

    template <typename Derived>
    void update_xaux(const Eigen::MatrixBase<Derived> &Y){

        Mat exp_logc = column_topic.log_mean().unaryExpr(exp_op).transpose();
        Mat exp_logr = row_topic.log_mean().unaryExpr(exp_op);
        Mat exp_log = exp_logr * exp_logc;
        x_aux = Y.array() / exp_log.array() ;

    }

    template <typename Derived>
    void update_column_topic(const Eigen::MatrixBase<Derived> &Y)
    {

        Mat exp_logc = column_topic.log_mean().unaryExpr(exp_op);
        Mat exp_logr = row_topic.log_mean().unaryExpr(exp_op);

        //a
        Mat aux_r_c =  (x_aux.transpose() * exp_logr).array() * exp_logc.array();


        //b
        Mat row_sum = row_topic.mean().colwise().sum().transpose();


        // std::cout << row_sum << std::endl;
        // std::cout << "---" << std::endl;
        // std::cout << "---" << std::endl;
        // std::cout << "---" << std::endl;


        Mat row_sum_tcol = (row_sum * onesN.transpose()).transpose();

        // std::cout << row_sum_tcol << std::endl;
        // std::cout << "---" << std::endl;

        column_topic.update( aux_r_c,row_sum_tcol);

        column_topic.calibrate();
    }

    template <typename Derived>
    void update_row_topic(const Eigen::MatrixBase<Derived> &Y)
    {
        Mat exp_logc = column_topic.log_mean().unaryExpr(exp_op);
        Mat exp_logr = row_topic.log_mean().unaryExpr(exp_op);

        //a
        Mat aux_c_r =(x_aux * exp_logc).array() * exp_logr.array();

        //b
        Mat col_sum = column_topic.mean().colwise().sum().transpose();

        // std::cout << col_sum << std::endl;
        // std::cout << "---" << std::endl;
        // std::cout << "---" << std::endl;
        // std::cout << "---" << std::endl;



        Mat col_sum_trow = (col_sum * onesD.transpose()).transpose();

        // std::cout << col_sum_trow << std::endl;
        // std::cout << "---" << std::endl;

        row_topic.update( aux_c_r, col_sum_trow);

        row_topic.calibrate();

    }

    const Index D, N, K;

    T onesD;
    T onesN;

    RNG rng;

    PARAM row_topic;
    PARAM column_topic;
    T x_aux;

    struct log_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterlog(x); }
    } log_op;

    struct exp_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterexp(x); }
    } exp_op;
};

#endif
