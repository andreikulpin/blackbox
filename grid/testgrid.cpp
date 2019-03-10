#include <iostream>
#include <iterator>
#include "gridsolver.hpp"

constexpr int n = 3;

double f(const double* x) {
	double v = 0.0;
	for (int i = 0; i < n; i++)
		v += x[i] * x[i];
	return v;
}
int main() {
	GridSolver<double> gs;
	double x[n];
	double a[n], b[n];
	std::fill(a, a + n, -1);
	std::fill(b, b + n, 2);
	gs.setparams(5, 0.1);
	double v = gs.search(n, x, a, b, f);
	gs.checkErrors(std::cerr);
	std::cout << "Found " << v << " at [";
	std::copy(x, x + n, std::ostream_iterator<double>(std::cout, " "));
	std::cout << "]\n";
	return 0;
}
