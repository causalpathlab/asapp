

// ----------------
// Regular C++ code
// ----------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../cpp/coordinatedescent.h"
#include "../cpp/gradientdescent.h"
#include "../cpp/mtxio.h"
#include <iostream>
#include <functional>

double ObjectFcn(std::vector<double> *funcLoc)
{
	double x = funcLoc->at(0);
	double y = funcLoc->at(1);

	return (x * x) + (x*y) + (y*y);
}


std::vector<double> rungd(std::vector<double> lst)
{
	// Create a function pointer for our object function.
	std::function<double(std::vector<double>*)> p_ObjectFcn{ ObjectFcn };
	
	// Create a test instance of the GradientDescent class.
	gradientdescent solver;
	
	// Assign the object function.
	solver.SetObjectFcn(p_ObjectFcn);
	
	// Set a start point.
	std::vector<double> startPoint = lst;
	solver.SetStartPoint(startPoint);
	
	// Set the maximum number of iterations.
	solver.SetMaxIterations(50);
	
	// Set the step size.
	solver.SetStepSize(0.1);
	
	// Call optimize.
	std::vector<double> funcLoc;
	double funcVal;
	solver.Optimize(&funcLoc, &funcVal);
	
	// Output the result.
	std::cout << "Function location: " << funcLoc[0]<<funcLoc[1] << std::endl;
	std::cout << "Function value: " << funcVal << std::endl;

  return funcLoc;

};

std::vector<double> runcd (char* xname, char* yname ){


	Eigen::MatrixXd mat_A = loadMtxToMatrix(xname);
	Eigen::MatrixXd mat_y = loadMtxToMatrix(yname);

	double lambda = 0.1;
	int epoch = 2;
	bool intercept = true;

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

	Eigen::VectorXd x = lasso(mat_A, vec_y, lambda, intercept,epoch); 

	std::vector<double> x2(x.data(), x.data() + x.size());

	return x2;

}
// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(scglm,m)
{
  m.doc() = "pybind11 example plugin";
  m.def("rungd", &rungd, "gradient descent");
  m.def("runcd", &runcd, "coordinate descent");
}