/* 
 * File:   testbenchros.cpp
 * Author: andrei
 */

#include <iostream>
#include "rosenbrockmethod.hpp"
#include "math.h"

#include "testfuncs/manydim/benchmarks.hpp"
#include <oneobj/contboxconstr/benchmarkfunc.hpp>

using BM = Benchmark<double>;
using namespace std;

bool testBench(std::shared_ptr<BM> bm, double eps) {
    const int dim = bm->getDim();
    
    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();
    
    LOCSEARCH::RosenbrockMethod<double> searchMethod;
    searchMethod.getOptions().mHInit = std::vector<double>({1., 1.});
    searchMethod.getOptions().mDoTracing = false;
    searchMethod.getOptions().mDoOrt = true;
    searchMethod.getOptions().mMaxStepsNumber = 10000;
    searchMethod.getOptions().mMinGrad = 1e-3;
    searchMethod.getOptions().mHLB = searchMethod.getOptions().mMinGrad * 1e-2;
    
    double a[dim], b[dim];
    double x[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first;
        b[i] = bm->getBounds()[i].second;
        x[i] = (b[i] + a[i]) / 2.0;
    }
    
    std::function<double (const double*) > func = [&] (const double * x) {
        return mpp->mObjectives.at(0)->func(x);
    };

    double result = searchMethod.search(dim, x, a, b, func);
    
    std::cout << bm->getDesc() << "\t";
    std::cout /*<< "Glob. min. = " */<< bm->getGlobMinY() << "\t";
    std::cout /*<< "Found value = " */<< result << "\t";
    std::cout /*<< "Iterations = " */<< searchMethod.getIterationsCount() << "\t";
    std::cout /*<< "Fun. Calls count = " */<< searchMethod.getFunctionCallsCount() << "\t" << "\n";
    return true;
}

int main(int argc, char** argv) {
    const double eps = argc > 2 ? atof(argv[2]) : 0.01;
    const int dim = argc > 1 ? atoi(argv[1]) : 2;

    /*auto bm = std::make_shared<RosenbrockBenchmark<double>>(dim);
    testBench(bm, eps);*/

    /*auto bm = std::make_shared<KeaneBenchmark<double>>();
    testBench(bm, eps);*/

    /*auto bm = std::make_shared<Ackley2Benchmark<double>>(4);
    testBench(bm, eps);*/

    auto bm = std::make_shared<DixonPriceBenchmark<double>>();
    testBench(bm, eps);

    /*Benchmarks<double> tests;
    for (auto bm : tests) {
        testBench(bm, eps);
    }*/
    return 0;
}

