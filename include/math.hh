#include "fastexp.h"
#include "fastlog.h"
#include "fastgamma.h"

#ifndef _MATH_HH_
#define _MATH_HH_

/////////////////////
// log(1 + exp(x)) //
/////////////////////

template <typename T>
inline T
_softplus(const T x)
{
    const T cutoff = static_cast<T>(10.);
    const T one = static_cast<T>(1.0);
    if (x > cutoff) {
        return x + fasterlog(one + fasterexp(-x));
    }
    return fasterlog(one + fasterexp(x));
}

/////////////////////
// 1/(1 + exp(-x)) //
/////////////////////

template <typename T>
inline T
_sigmoid(const T x, const T pmin = 0.0, const T pmax = 1.0)
{
    const T cutoff = static_cast<T>(10.);
    const T one = static_cast<T>(1.0);

    if (x < cutoff) {
        return pmax * fasterexp(x) / (one + fasterexp(x));
    }

    return pmax / (one + fasterexp(-x)) + pmin;
}

//////////////////////////
// log(exp(a) + exp(b)) //
//////////////////////////

template <typename T>
inline T
_log_sum_exp(const T log_a, const T log_b)
{
    if (log_a > log_b) {
        return log_a + _softplus(log_b - log_a);
    }
    return log_b + _softplus(log_a - log_b);
}

///////////////////////////////////////
// modified Bessel of the first kind //
///////////////////////////////////////

template <typename IDX, typename F>
inline F
_log_bessel_i(const IDX p, const F x)
{

    // We compute log Bessel function by the log-sum-exp trick
    //
    // log Ip(x) = p log(x/2) + log sum_j f(x,j)
    // where
    // log f(x,j) = 2j log(x/2) - lgamma(j + 1) - lgamma(p + j + 1)
    //
    // Note: log f(x,0) = -lgamma(p + 1)
    //

    const F _p = static_cast<F>(p);

    F N = static_cast<F>(3.0 * p);
    const F log_x_half = fasterlog(x * 0.5);

    F log_sum_series = fasterlgamma(_p + 1.0);

    for (F j = 1; j < N; ++j) {

        const F _log_f = 2.0 * j * log_x_half - fasterlgamma(j + 1.0) -
            fasterlgamma(_p + j + 1.0);

        log_sum_series = _log_sum_exp(log_sum_series, _log_f);
    }

    return log_sum_series + _p * log_x_half;
}

#endif
