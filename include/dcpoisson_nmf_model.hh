#include "svd.hh"
#include "eigen_util.hh"
#include "mmutil.hh"

#ifndef DCPOISSON_NMF_MODEL_HH_
#define DCPOISSON_NMF_MODEL_HH_

template <typename T, typename RNG, typename PARAM>
struct dcpoisson_nmf_t {

    using Scalar = typename T::Scalar;
    using Index = typename T::Index;

    explicit dcpoisson_nmf_t(const Index d,
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
        , row_degree(d, 1, a0, b0, rng)
        , column_degree(n, 1, a0, b0, rng)
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
        column_degree.initialize();
        column_topic.initialize();
        row_degree.initialize();
        row_topic.initialize();
    }

    template <typename Derived>
    void update_degree(const Eigen::MatrixBase<Derived> &Y)
    {
        column_degree.update(Y.transpose() * onesD,
                                  onesN * row_degree.mean().sum());
        column_degree.calibrate();

        row_degree.update(Y * onesN,
                               onesD * column_degree.mean().sum());
        row_degree.calibrate();

    }

    void update_degree_baseline()
    {
        const Scalar row_sum = row_degree.mean().sum();
        row_degree.baseline_div(row_sum);
        column_degree.baseline_mult(row_sum);
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
        Mat r_topic_degree = row_topic.mean().transpose() * row_degree.mean();
        std::cout << "---" << std::endl;        
        std::cout << "---" << std::endl;        
        std::cout << row_topic.mean().transpose() << std::endl;
        std::cout << "---" << std::endl;
        
        std::cout << row_degree.mean() << std::endl;
        
        std::cout << "---" << std::endl;
        std::cout << r_topic_degree << std::endl;

        Mat r_topic_degree_ns = (r_topic_degree * onesN.transpose()).transpose();

        std::cout << "---" << std::endl;
        std::cout << r_topic_degree_ns << std::endl;

        std::cout << "---" << std::endl;
        
        Mat a = column_degree.mean().asDiagonal();
        std::cout << a << std::endl;

        std::cout << "---" << std::endl;
        Mat b = column_degree.mean().asDiagonal() * r_topic_degree_ns;
        std::cout << b << std::endl;


        std::cout << "---" << std::endl;
        std::cout << "---" << std::endl;


        Mat t = column_degree.mean().asDiagonal() * r_topic_degree_ns;
        column_topic.update( aux_r_c,column_degree.mean().asDiagonal() * r_topic_degree_ns);

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
        Mat c_topic_degree = column_topic.mean().transpose() * column_degree.mean();
        Mat c_topic_degree_ds = (c_topic_degree * onesD.transpose()).transpose();
        
        row_topic.update( aux_c_r,row_degree.mean().asDiagonal() * c_topic_degree_ds);

        row_topic.calibrate();

    }

    const Index D, N, K;

    T onesD;
    T onesN;

    RNG rng;

    PARAM row_degree;
    PARAM column_degree;

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
