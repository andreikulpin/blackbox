/* 
 * File:   testmodelbasedmethod.cpp
 * Author: andrei
 *
 * Created on January 20, 2019, 5:44 PM
 */

#include <iostream>
#include "modelbasedmethod.hpp"
#include "math.h"

#include "testfuncs/manydim/benchmarks.hpp"
#include <oneobj/contboxconstr/benchmarkfunc.hpp>

using BM = Benchmark<double>;
using namespace std;

bool testBench(std::shared_ptr<BM> bm, double eps) {
    const int dim = bm->getDim();
    
    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();
    
    LOCSEARCH::ModelBasedMethod<double> searchMethod;
    searchMethod.getOptions().mDoTracing = false;
    searchMethod.getOptions().useDogleg = true;
    searchMethod.getOptions().mMaxIterations = 500;
    searchMethod.getOptions().mEps = 1e-8;
    searchMethod.getOptions().mMinGrad = eps;
    searchMethod.getOptions().mFunctionGlobMin = bm->getGlobMinY();
    searchMethod.getOptions().mDoSavingPath = false;
    
    double a[dim], b[dim];
    double initX[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first ;
        b[i] = bm->getBounds()[i].second;
        initX[i] = (b[i] + a[i]) / 2.0;
    }
    
    std::function<double (const double*) > func = [&] (const double * x) {
        return mpp->mObjectives.at(0)->func(x);
    };

    double result = searchMethod.search(dim, initX, a, b, func);
    
    std::cout << bm->getDesc() << "\t";
    std::cout /*<< "Glob. min. = " */<< bm->getGlobMinY() << "\t";
    std::cout /*<< "Glob. min. x = " */ << snowgoose::VecUtils::vecPrint(dim, bm->getGlobMinX().data()) << "\t";
    std::cout /*<< "Found value = " */<< result << "\t";
    std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(dim, initX) << "\t";
    std::cout /*<< "Iterations = " */<< searchMethod.getIterationsCount() << "\t";
    std::cout /*<< "Fun. Calls count = " */<< searchMethod.getFunctionCallsCount() << "\t" << "\n";
    return true;
}

int main(int argc, char** argv) {
    const int dim = argc > 1 ? atoi(argv[1]) : 2;
    const double eps = argc > 2 ? atof(argv[2]) : 1e-10;

    auto bm = std::make_shared<RosenbrockBenchmark<double>>(dim);
    testBench(bm, eps);

    /*Benchmarks<double> tests;
    for (auto bm : tests) {
        testBench(bm, eps);
    }*/
    return 0;
}

