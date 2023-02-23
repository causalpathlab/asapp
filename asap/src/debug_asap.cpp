#include <iostream>
#include "../include/cpp_asap.hh"

int main()
{
Eigen::MatrixXf mat = Eigen::MatrixXf::Random(20, 10);

int maxk = 10;

ASAPNMF model(mat,maxk);
model.nmf();

std::cout<<"ok"<<std::endl;
return 0;
}