#include <iostream>
#include <cstdlib>
#include "best_point_method.hpp"


using namespace std;

double func(const double* x) {
    return 100 * SGSQR(x[1] - x[0] * x[0]) + SGSQR(1 - x[1]);
}

int main(int argc, char** argv) {
    const int dim = 2;

    double x[dim] = {3, 3};
    
    double a[dim], b[dim];
    std::fill(a, a + dim, -4);
    std::fill(b, b + dim, 8);

    LOCSEARCH::BestPointMethod<double> searchMethod;
    searchMethod.getOptions().mHInit = std::vector<double>({1., 1.});
    searchMethod.getOptions().mDoTracing = true;
    searchMethod.getOptions().mMaxStepsNumber = 10000;
    searchMethod.getOptions().mInc = 1.418;
    searchMethod.getOptions().mDec = 0.368;
    searchMethod.getOptions().numbOfPoints = 100;
    double v = searchMethod.search(dim, x, a, b, func);

    std::cout << searchMethod.about() << "\n";
    std::cout << "Found v = " << v << "\n";
    std::cout << " at " << snowgoose::VecUtils::vecPrint(dim, x) << "\n";
    return 0;
}

