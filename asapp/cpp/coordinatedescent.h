
#ifndef COORDINATEDESCENT_H
#define COORDINATEDESCENT_H

#include <eigen3/Eigen/Dense>

Eigen::VectorXd lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          bool intercept,
          int iter_max);

#endif