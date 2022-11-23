#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "rpstruct.h"
#include "mtxio.h"
#include "mmio.h"


int main()
{

Eigen::MatrixXd mat_A = loadMtxToMatrix("../../../data/test_x.mtx");
Eigen::MatrixXd mat_y = loadMtxToMatrix("../../../data/test_y.mtx");


split(mat_A);

std::cout<<"ok"<<std::endl;
return 0;
}