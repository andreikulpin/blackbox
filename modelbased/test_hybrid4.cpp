/* 
 * File:   testmodelbasedmethod.cpp
 * Author: andrei
 *
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include "modelbasedmethod.hpp"
#include "../rosenbrock/rosenbrockmethod.hpp"
#include "math.h"

#include "testfuncs/manydim/benchmarks.hpp"
#include <oneobj/contboxconstr/benchmarkfunc.hpp>

using BM = Benchmark<double>;
using namespace std;

int testMB(std::shared_ptr<BM> bm, double eps, double * xModelBased, double * a, double * b, const char * pathFile) {
    const int dim = bm->getDim();
    
    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();
    
    LOCSEARCH::ModelBasedMethod<double> modelBasedMethod;
    modelBasedMethod.getOptions().mDoTracing = false;
    modelBasedMethod.getOptions().useDogleg = true;
    modelBasedMethod.getOptions().mMaxIterations = 1000;
    modelBasedMethod.getOptions().mEps = eps * 1;
    modelBasedMethod.getOptions().mMinGrad = eps * 1e-2;
    modelBasedMethod.getOptions().mFunctionGlobMin = bm->getGlobMinY();
    modelBasedMethod.getOptions().mDoSavingPath = true;
    modelBasedMethod.getOptions().pathFile = pathFile;

    std::function<double (const double*) > func = [&] (const double * xModelBased) {
        return mpp->mObjectives.at(0)->func(xModelBased);
    };

    double resultModelBased = modelBasedMethod.search(dim, xModelBased, a, b, func);
    return modelBasedMethod.getFunctionCallsCount();
}

int testRosenbrock(std::shared_ptr<BM> bm, double eps, double * xRosenbrock, double * a, double * b, std::vector<double> &rosPath, std::vector<int> &rosCallsCounts) {
    const int dim = bm->getDim();
    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();

    std::function<double (const double*) > func = [&] (const double * xModelBased) {
        return mpp->mObjectives.at(0)->func(xModelBased);
    };

    LOCSEARCH::RosenbrockMethod<double> rosenbrockMethod;
    rosenbrockMethod.getOptions().mHInit = std::vector<double>({1., 1.});
    rosenbrockMethod.getOptions().mDoTracing = false;
    rosenbrockMethod.getOptions().mDoOrt = true;
    rosenbrockMethod.getOptions().mMaxStepsNumber = 1000;
    rosenbrockMethod.getOptions().mMinGrad = eps * 1e-2;
    rosenbrockMethod.getOptions().mHLB = 1e-6;
    rosenbrockMethod.getOptions().mFunctionGlobMin = bm->getGlobMinY();
    rosenbrockMethod.getOptions().mEps = eps * 1;

    double resultRosenbrock = rosenbrockMethod.search(dim, xRosenbrock, a, b, func);

    /*std::cout << "\nRos iters = " << rosenbrockMethod.getIterationsCount() << " ";
    std::cout << "Ros path size = " << rosenbrockMethod.getPath().size() << " ";
    std::cout << "Ros call counts size = " << rosenbrockMethod.getCallCounts().size() << " ";*/

    rosPath.assign(rosenbrockMethod.getPath().begin(), rosenbrockMethod.getPath().end());
    rosCallsCounts.assign(rosenbrockMethod.getCallCounts().begin(), rosenbrockMethod.getCallCounts().end());

    // std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(dim, xRosenbrock) << "\t";
    // std::cout /*<< "Found value = " */<< resultRosenbrock << "\t";
    // std::cout /*<< "Iterations = " */<< rosenbrockMethod.getIterationsCount() << "\t";
    // std::cout /*<< "Fun. Calls count = " */<< rosenbrockMethod.getFunctionCallsCount() << "\t";
    // std::cout /*<< "Fun. Calls Needed = " */<< rosenbrockMethod.getFunctionCallsNeeded() << "\t\n";

    return rosenbrockMethod.getFunctionCallsCount();
}

bool testWithInitialPoint(std::shared_ptr<BM> bm, double eps, double x, double y, bool isFirstInitPoint) {
    
    const int dim = bm->getDim();
    double a[dim], b[dim];
    double xInit[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first ;
        b[i] = bm->getBounds()[i].second;
        xInit[i] = (b[i] + a[i]) / 2.0;
    }

    xInit[0] = x;
    xInit[1] = y;

    double xRos[dim];
    snowgoose::VecUtils::vecCopy(dim, xInit, xRos);

    std::vector<double> rosPath;
    std::vector<int> rosCallCounts;
    std::vector<int> mbCallCounts;
    int rosCalls = testRosenbrock(bm, eps, xRos, a, b, rosPath, rosCallCounts);

    int takenRosCalls;
    double xModelBased[dim];
    double z;
    bool isFirstPoint = true;

    snowgoose::VecUtils::vecCopy(dim, &(rosPath[dim * 0]), xModelBased);
    int mbCalls = testMB(bm, eps, xModelBased, a, b, "path.log");

    int index = 1;
    snowgoose::VecUtils::vecCopy(dim, &(rosPath[dim * index]), xModelBased);
    int hybridMbCalls = testMB(bm, eps, xModelBased, a, b, "pathh.log");
    int hybridCalls = hybridMbCalls + rosCallCounts[index];

    std::cout << bm->getDesc() << "\t";
    std::cout << snowgoose::VecUtils::vecPrint(dim, xInit) << "\t";
    std::cout << rosCalls << "\t";
    std::cout << mbCalls << "\t";
    std::cout << hybridCalls << "\t";
    std::cout << "\n";

    rosPath.clear();
    rosCallCounts.clear();
    mbCallCounts.clear();

    return true;
}

int main(int argc, char** argv) {
    //const int dim = argc > 1 ? atoi(argv[1]) : 2;
    const double eps = argc > 3 ? atof(argv[3]) : 1e-4;

    const double x = argc > 1 ? atof(argv[1]) : 0;
    const double y = argc > 2 ? atof(argv[2]) : 0;

    bool isFirstBenchmark = true;

    std::cout << std::fixed << std::setprecision(12);

    auto bm = std::make_shared<Price2Benchmark<double>>();
    testWithInitialPoint(bm, eps, x, y, true);

    /*std::shared_ptr<Benchmark<double>> bms[3] = {
        std::make_shared<Ackley2Benchmark<double>>(3),
        std::make_shared<BartelsConnBenchmark<double>>(),
        std::make_shared<Alpine2Benchmark<double>>()
    };

    for (auto bm : bms) {
        testBench(bm, eps, x, y, true); 
        isFirstBenchmark = false;
    }*/

    /*Benchmarks<double> tests;
    for (auto bm : tests) {
        testBench(bm, eps, x, y, isFirstBenchmark);
        isFirstBenchmark = false;
    }*/

    //std::cout << "\n Tested " << "\n";
    return 0;
}

