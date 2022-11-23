#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <eigen3/Eigen/Dense>

#include "mtxio.h"
#include "coordinatedescent.h"

void usage(char* argv[]){
    printf("%s [sample] [label] [lambda] [stepsize] [batch_size] [epoch]\n", argv[0]);
}


int main(int argc, char* argv[])
{
    if (argc != 7){
        usage(argv);
        return 1;
    }
    
    char* A_file_name = argv[1];
    char* y_file_name = argv[2];
    double lambda = atof(argv[3]);
    double stepsize = atof(argv[4]);
    int batch_size = atoi(argv[5]);
    int epoch = atoi(argv[6]);

    std::cout <<"processing..."<<std::endl;

    Eigen::MatrixXd mat_A = loadMtxToMatrix(A_file_name);
    Eigen::MatrixXd mat_y = loadMtxToMatrix(y_file_name);

    // center and normalize
    Eigen::VectorXd mat_A_mean = mat_A.colwise().mean();
    mat_A.rowwise() -= mat_A_mean.transpose();

    // normalize columns of matrix A
    Eigen::VectorXd A_norm = mat_A.colwise().norm();
    for (int i = 0; i < A_norm.size(); i++) {
        if (A_norm(i) == 0.0)
            A_norm(i) = 1.0;
        else 
            A_norm(i) = 1.0 / A_norm(i);
    }
    
    mat_A = mat_A * (A_norm.asDiagonal());

    std::cout <<"processed"<<std::endl;

    //map mat_y into a vector
    Eigen::VectorXd vec_y(Eigen::Map<Eigen::VectorXd>(mat_y.data(), mat_y.cols()*mat_y.rows()));

    bool intercept = true; 

    Eigen::VectorXd x = lasso(mat_A, vec_y, lambda, intercept,epoch); 

    std::cout << "coefficients: " << std::endl;
    std::cout << x << std::endl;

    return 0;
}
