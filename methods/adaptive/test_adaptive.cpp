#include <iostream>
#include <cstdlib>
#include "adaptive_method.hpp"

using namespace std;

double func(const double* x) {
    return 100 * SGSQR(x[1] - x[0] * x[0]) + SGSQR(1 - x[1]);
}

/*double func(const double* x) {
            return 4 * SGSQR(x[0] - 5) + SGSQR(x[1] - 6);
        }*/

int main(int argc, char** argv) {
    const int dim = 2;

    double x[dim] = {3, 3};
    
    double a[dim], b[dim];
    std::fill(a, a + dim, -4);
    std::fill(b, b + dim, 8);

    LOCSEARCH::AdaptiveMethod<double> searchMethod;
    searchMethod.getOptions().mDoTracing = true;
    searchMethod.getOptions().maxStepNumber = 100;
    searchMethod.getOptions().mInc = 1.418;
    searchMethod.getOptions().mDec = 0.368;
    searchMethod.getOptions().numbOfPoints = 20;
    double v = searchMethod.search(dim, x, a, b, func);

    std::cout << searchMethod.about() << "\n";
    std::cout << "Found v = " << v << "\n";
    std::cout << " at " << snowgoose::VecUtils::vecPrint(dim, x) << "\n";
    return 0;
}
