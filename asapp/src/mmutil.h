#include <pybind11/eigen.h>

#ifndef MMUTIL_HH_
#define MMUTIL_HH_

using Scalar = float;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, std::ptrdiff_t>;
using Index = SpMat::Index;
using MSpMat = Eigen::MappedSparseMatrix<Scalar>;

using Mat = typename Eigen::
    Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using IntMat = typename Eigen::
    Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using IntVec = typename Eigen::Matrix<std::ptrdiff_t, Eigen::Dynamic, 1>;

#endif
