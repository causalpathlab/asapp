#include <iostream>
#include <string>
#include <cstdlib>

#include "mtxio.h"
#include "coordinatedescent.h"


inline double softThresholdingOperator(double rho, double lambda){
    if (rho < -lambda) return lambda + rho;
    else if (rho > lambda) return rho - lambda;
    else return 0;
} 

// solve the lasso problem by using coordinate descent
// ||Ax - y ||^2 + lambda*|x|
//
// Assumptions:
//  1. A: each row is a sample, and each column is a feature.
//  2. number of row of A is equal to the length of vector y
Eigen::VectorXd lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          bool intercept,
          int iter_max)
{
    // add one column to A if intercept is set to true
    if (intercept){
        A.conservativeResize(A.rows(), A.cols()+1); 
        for (int j=0; j < A.rows(); j++)
            A(j, A.cols()-1) = 1.0;
    }

    // initialize solution vector x, with 0.
    Eigen::VectorXd x(A.cols());
    x.setZero();  
    if (intercept) {
        x(x.size()-1) = y.sum() / y.size();
    }

    // main loop for descent
    for(int ite=0; ite < iter_max; ite++) {

        for (int idx = 0; idx < x.size()-1; idx++) {
            Eigen::VectorXd r = y - A*x;     
            double rho = A.col(idx).transpose()* (r + x(idx) *A.col(idx));
            x(idx) = softThresholdingOperator(rho, lambda);
        }

        // this second formula also works
        //following the trick in lecture slides of "coordinate descent"
        //from Geoff Gordon & Ryan Tibshirani at P10.
        //Eigen::VectorXd r = y - A*x;     

        // for (int idx = 0; idx < x.size()-1; idx++) {
        //     Eigen::VectorXd r = y - A*x;     
        //     double rho = A.col(idx).transpose()*r + x(idx);
        //     x(idx) = softThresholdingOperator(rho, lambda);
        // }

        if (intercept){
            x(x.size()-1) = (y-A*x).sum()/A.rows();
        }       

        //info
        Eigen::VectorXd r = y - A*x;     
        std::cout <<"ite: " << ite <<" residual: "<< r.norm() << std::endl;

    }
        
    return x;
}
