#include <iostream>
#include "../include/cpp_asap.hh"

int main()
{
Eigen::MatrixXf mat = Eigen::MatrixXf::Random(4, 3);

int maxk = 2;

// ASAPNMFAlt model(mat,maxk);
// model.nmf();

Eigen::MatrixXf y = Eigen::MatrixXf::Random(4, 3);
Eigen::MatrixXf beta = Eigen::MatrixXf::Random(4, 2);

ASAPREG model(y,beta);
model.regression();

std::cout<<"ok"<<std::endl;
return 0;
}