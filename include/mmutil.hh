#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Core>

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


using IntegerMatrix = typename Eigen::
        Matrix<std::ptrdiff_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using RowVec = typename Eigen::internal::plain_row_type<Mat>::type;

#endif
