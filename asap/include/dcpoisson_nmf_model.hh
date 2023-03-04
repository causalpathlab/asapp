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
        , topic_loading(k, 1, a0, b0, rng)
        , x_aux(d,n)
    {
        onesD.setOnes();
        onesN.setOnes();
        x_aux.setOnes();
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

    //TODO dc log likelihood 

    //TODO update this log likelihood for dc version

    template <typename Derived, typename Latent>
    const Scalar log_likelihood(const Eigen::MatrixBase<Derived> &Y,
                                const Latent &latent)
    {
        double term0 = 0., term1 = 0., term2 = 0.;

        term0 += (Y.array().colwise() * take_row_log_degree().array()).sum();

        term0 +=
            (Y.array().rowwise() * take_column_log_degree().transpose().array())
                .sum();

        for (Index k = 0; k < K; ++k) {

            term1 += latent.slice_k(Y, k).sum() * take_topic_log_loading(k);

            term1 += (latent.slice_k(Y, k).array().colwise() *
                      row_topic.log_mean().col(k).array())
                         .sum();
            term1 += (latent.slice_k(Y, k).array().rowwise() *
                      column_topic.log_mean().col(k).transpose().array())
                         .sum();
        }

        term2 =
            (((row_topic.mean().array().colwise() * take_row_degree().array())
                  .rowwise() *
              take_topic_loading().transpose().array())
                 .matrix() *
             (column_topic.mean().transpose().array().rowwise() *
              take_column_degree().transpose().array())
                 .matrix())
                .sum();

        return term0 + term1 - term2;
    }

    inline const auto take_row_degree() const
    {
        return row_degree.mean().col(0);
    }

    inline const auto take_topic_loading() const
    {
        return topic_loading.mean().col(0);
    }

    inline const auto take_topic_log_loading() const
    {
        return topic_loading.log_mean().col(0);
    }

    inline Scalar take_topic_loading(const Index k) const
    {
        return topic_loading.mean().coeff(k, 0);
    }

    inline Scalar take_topic_log_loading(const Index k) const
    {
        return topic_loading.log_mean().coeff(k, 0);
    }

    inline const auto take_row_log_degree() const
    {
        return row_degree.log_mean().col(0);
    }

    inline const auto take_column_degree() const
    {
        return column_degree.mean().col(0);
    }

    inline const auto take_column_log_degree() const
    {
        return column_degree.log_mean().col(0);
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
        Mat aux_r = x_aux.transpose() * exp_logr;
        Mat aux_r_c = aux_r.array() * exp_logc.array();

        //b
        Mat r_topic_degree = row_topic.mean().transpose() * row_degree.mean();
        Mat r_topic_degree_ns = (r_topic_degree * onesN.transpose()).transpose();
        
        column_topic.update( aux_r_c,column_degree.mean().asDiagonal() * r_topic_degree_ns);

        column_topic.calibrate();
    }

    template <typename Derived>
    void update_row_topic(const Eigen::MatrixBase<Derived> &Y)
    {
        Mat exp_logc = column_topic.log_mean().unaryExpr(exp_op);
        Mat exp_logr = row_topic.log_mean().unaryExpr(exp_op);

        //a
        Mat aux_c = x_aux * exp_logc;
        Mat aux_c_r = aux_c.array() * exp_logr.array();

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
    PARAM topic_loading;

    T x_aux;

    struct log_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterlog(x); }
    } log_op;

    struct log1p_op_t {
        const Scalar operator()(const Scalar &x) const
        {
            return fasterlog(1. + x);
        }
    } log1p_op;

    struct exp_op_t {
        const Scalar operator()(const Scalar &x) const { return fasterexp(x); }
    } exp_op;
};

#endif
