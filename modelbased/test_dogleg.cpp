/* 
 * File:   testmodelbasedmethod.cpp
 * Author: andrei
 *
 */

#include <iostream>
#include <iomanip>
#include "modelbasedmethod.hpp"
#include "math.h"

#include "testfuncs/manydim/benchmarks.hpp"
#include <oneobj/contboxconstr/benchmarkfunc.hpp>

using BM = Benchmark<double>;
using namespace std;

int testMB(std::shared_ptr<BM> bm, double eps, double * xModelBased, double * a, double * b, bool useDogleg, double & res, int & callsNeeded) {
    const int dim = bm->getDim();
    
    OPTITEST::BenchmarkProblemFactory problemFactory(bm);
    COMPI::MPProblem<double> *mpp = problemFactory.getProblem();
    
    LOCSEARCH::ModelBasedMethod<double> modelBasedMethod;
    modelBasedMethod.getOptions().mDoTracing = false; 
    modelBasedMethod.getOptions().useDogleg = useDogleg;
    modelBasedMethod.getOptions().mMaxIterations = 1000;
    modelBasedMethod.getOptions().mEps = eps * 1;
    modelBasedMethod.getOptions().mMinGrad = eps * 1e-2;
    modelBasedMethod.getOptions().mFunctionGlobMin = bm->getGlobMinY();
    modelBasedMethod.getOptions().mDoSavingPath = true;

    if (useDogleg) {
         modelBasedMethod.getOptions().pathFile = "pathd.log";
    }

    std::function<double (const double*) > func = [&] (const double * xModelBased) {
        return mpp->mObjectives.at(0)->func(xModelBased);
    };

    double resultModelBased = modelBasedMethod.search(dim, xModelBased, a, b, func);
    res = resultModelBased;
    callsNeeded = modelBasedMethod.getFunctionCallsNeeded();
    return modelBasedMethod.getFunctionCallsCount();
}

void testWithInitialPoint(std::shared_ptr<BM> bm, double * a, double * b, double eps, double x, double y, bool isFirstBenchmark) {
    const int dim = bm->getDim();
    double xInit[dim];
    for (int i = 0; i < dim; i++) {
        xInit[i] = (b[i] + a[i]) / 2.0;
    }

    xInit[0] = x; 
    xInit[1] = y;

    double xMB[dim];
    double xMBDogleg[dim];

    snowgoose::VecUtils::vecCopy(dim, xInit, xMB);
    snowgoose::VecUtils::vecCopy(dim, xInit, xMBDogleg);

    double resMB;
    double resMBDogleg;

    int mbCallsNeeded = 0;
    int mbDoglegCallsNeeded = 0;

    int mbCalls = testMB(bm, eps, xMB, a, b, false, resMB, mbCallsNeeded);
    int mbDoglegCalls = testMB(bm, eps, xMBDogleg, a, b, true, resMBDogleg, mbDoglegCallsNeeded);

    bool isResultsClose = snowgoose::VecUtils::vecDistAbs(dim, xMB, xMBDogleg) < 1e-2;
    if (isResultsClose) {
        std::cout << bm->getDesc() << "\t"; 
        std::cout << snowgoose::VecUtils::vecPrint(dim, xInit) << "\t";
        std::cout << snowgoose::VecUtils::vecPrint(dim, xMB) << "\t";
        std::cout << snowgoose::VecUtils::vecPrint(dim, xMBDogleg) << "\t";
        std::cout << resMB << "\t";
        std::cout << resMBDogleg << "\t";
        std::cout << mbCalls << "\t";
        std::cout << mbDoglegCalls << "\t";
        std::cout << mbCallsNeeded << "\t";
        std::cout << mbDoglegCallsNeeded << "\t";
        std::cout << (mbCalls >= mbDoglegCalls ? "true" : "false") << "\t" << "\n";
        //std::cout << (isResultsClose ? "true" : "false") << "\n";
    }
}

void testBench(std::shared_ptr<BM> bm, double eps, double x, double y, bool isFirstBenchmark) {
    const int dim = bm->getDim();

    //if (dim > 2) return;

    double a[dim], b[dim];
    double xInit[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first ;
        b[i] = bm->getBounds()[i].second;
        //xInit[i] = (b[i] + a[i]) / 2.0;
    }

    bool isFirstLaunch = isFirstBenchmark;

    /*std::cout << "Benchmark\t" << "Glob min x\t" << "Glob min\n";  
    std::cout << bm->getDesc() << "\t";
    std::cout << snowgoose::VecUtils::vecPrint(dim, bm->getGlobMinX().data()) << "\t";
    std::cout << bm->getGlobMinY() << "\n";*/

    int steps = 1;
    double stepX = (bm->getBounds()[0].second - bm->getBounds()[0].first) / (steps + 1);
    double stepY = (bm->getBounds()[1].second - bm->getBounds()[1].first) / (steps + 1);
    //std::cout << bm->getBounds()[0].first << ":" << bm->getBounds()[0].second << " " << stepX << "\n";
    //std::cout << bm->getBounds()[1].first << ":" << bm->getBounds()[1].second << " " << stepY << "\n";
    for (int i = 1; i <= steps; i++) {  
        for (int j = 1; j <= steps; j++) {
            xInit[0] = bm->getBounds()[0].first + stepX * i;
            xInit[1] = bm->getBounds()[1].first + stepY * j;
            //std::cout << snowgoose::VecUtils::vecPrint(dim, xInit) << " \n";
            testWithInitialPoint(bm, a, b, eps, xInit[0], xInit[1], isFirstLaunch);
            isFirstLaunch = false;
        }
    }

    /*xInit[0] = x;
    xInit[1] = y;
    testWithInitialPoint(bm, a, b, eps, xInit[0], xInit[1], isFirstLaunch);*/

    //std::cout << "\n\n\n";
}

int main(int argc, char** argv) { 
    //const int dim = argc > 1 ? atoi(argv[1]) : 2;
    const double eps = argc > 3 ? atof(argv[3]) : 1e-4;

    const double x = argc > 1 ? atof(argv[1]) : 0;
    const double y = argc > 2 ? atof(argv[2]) : 0;

    bool isFirstBenchmark = true;

    std::cout << std::fixed << std::setprecision(12);
    
    /*auto bm = std::make_shared<RosenbrockBenchmark<double>>(2);
    testBench(bm, eps, x, y, true);*/

    /*auto bm = std::make_shared<GoldsteinPriceBenchmark<double>>();
    testBench(bm, eps);*/  
   
    /*auto bm = std::make_shared<VenterSobiezcczanskiSobieskiBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<KeaneBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<BartelsConnBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/   

    /*auto bm = std::make_shared<BradBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<QuadraticBenchmark<double>>();
    testBench(bm, eps, x, y, true);*/

    Benchmarks<double> tests; 
    for (auto bm : tests) { 
        testBench(bm, eps, x, y, isFirstBenchmark);
        isFirstBenchmark = false;
    }
    return 0;
}
