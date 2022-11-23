/* ************************************************************************
https://github.com/QuantitativeBytes
************************************************************************/

#include "gradientdescent.h"
#include <iostream>
#include <functional>

// Define the object function.
/* The object function can take any form and have any number of
	variables. Multiple variables are contained in the elements
	of the std::vector pointer funcLoc and access with:
	
	funcLoc->at(0);
	funcLoc->at(1);
	funcLoc->at(...);
	And so on.
	
	The function returns a single, double-precision floating
	point representing the value of the function at the location
	specified by the input variables. */
double ObjectFcn(std::vector<double> *funcLoc)
{
	double x = funcLoc->at(0);
	double y = funcLoc->at(1);

	return (x * x) + (x*y) + (y*y);
}

int main(int argc, char* argv[])
{
	// Create a function pointer for our object function.
	std::function<double(std::vector<double>*)> p_ObjectFcn{ ObjectFcn };
	
	// Create a test instance of the GradientDescent class.
	GradientDescent solver;
	
	// Assign the object function.
	solver.SetObjectFcn(p_ObjectFcn);
	
	// Set a start point.
	std::vector<double> startPoint = {5.0,5.0};
	solver.SetStartPoint(startPoint);
	
	// Set the maximum number of iterations.
	solver.SetMaxIterations(50);
	
	// Set the step size.
	solver.SetStepSize(0.1);
	
	std::vector<double> funcLoc;
	double funcVal;
	solver.Optimize(&funcLoc, &funcVal);
	
	std::cout << "Function location: " << funcLoc[0]<<funcLoc[1] << std::endl;
	std::cout << "Function value: " << funcVal << std::endl;
	
	return 0;
}
