/* 
 * File:   testmodelbasedmethod.cpp
 * Author: andrei
 *
 */

#include <iostream>
#include <iomanip>
#include "modelbasedmethod.hpp"
#include "../rosenbrock/rosenbrockmethod.hpp"
#include "math.h"

#include "testfuncs/manydim/benchmarks.hpp"
#include <oneobj/contboxconstr/benchmarkfunc.hpp>

using BM = Benchmark<double>;
using namespace std;

bool testWithInitialPoint(std::shared_ptr<BM> bm, double eps, double * initX, double * a, double * b, bool isFirstLaunch) {
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

    double xModelBased[dim];
    snowgoose::VecUtils::vecCopy(dim, initX, xModelBased);

    std::function<double (const double*) > func = [&] (const double * xModelBased) {
        return mpp->mObjectives.at(0)->func(xModelBased);
    };

    if (isFirstLaunch)
        std::cout << modelBasedMethod.about() << "\n";

    double resultModelBased = modelBasedMethod.search(dim, xModelBased, a, b, func);

    //bool isModelBasedConverged = snowgoose::VecUtils::vecDist(dim, xModelBased, bm->getGlobMinX().data()) < eps;
    bool isModelBasedConverged = std::abs(resultModelBased - bm->getGlobMinY()) < eps;
    
    double xRosenbrock[dim];
    snowgoose::VecUtils::vecCopy(dim, initX, xRosenbrock);

    LOCSEARCH::RosenbrockMethod<double> rosenbrockMethod;
    rosenbrockMethod.getOptions().mHInit = std::vector<double>({1., 1.});
    rosenbrockMethod.getOptions().mDoTracing = false;
    rosenbrockMethod.getOptions().mDoOrt = true;
    rosenbrockMethod.getOptions().mMaxStepsNumber = 1000;
    rosenbrockMethod.getOptions().mMinGrad = eps * 1e-2;
    rosenbrockMethod.getOptions().mHLB = 1e-6;
    rosenbrockMethod.getOptions().mFunctionGlobMin = bm->getGlobMinY();
    rosenbrockMethod.getOptions().mEps = eps * 1;

    if (isFirstLaunch)
        std::cout << rosenbrockMethod.about() << "\n";

    double resultRosenbrock = rosenbrockMethod.search(dim, xRosenbrock, a, b, func);
    //bool isRosenbrockConverged = snowgoose::VecUtils::vecDist(dim, xRosenbrock, bm->getGlobMinX().data()) < eps;
    bool isRosenbrockConverged = std::abs(resultRosenbrock - bm->getGlobMinY()) < eps;

    //if (isModelBasedConverged && isRosenbrockConverged) {
    bool isResultsClose = snowgoose::VecUtils::vecDistAbs(dim, xModelBased, xRosenbrock) < 1e-1;
    if (isResultsClose) {
        std::cout << bm->getDesc() << "\t";
        std::cout /*<< "Glob. min. = " */<< bm->getGlobMinY() << "\t";
        std::cout /*<< "Glob. min. x = " */ << snowgoose::VecUtils::vecPrint(dim, bm->getGlobMinX().data()) << "\t";
        std::cout /* << "Started from " */<< snowgoose::VecUtils::vecPrint(dim, initX) << "\t";
        std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(dim, xModelBased) << "\t";
        std::cout /*<< "Found value = " */<< resultModelBased << "\t";
        std::cout /*<< "Iterations = " */<< modelBasedMethod.getIterationsCount() << "\t";
        std::cout /*<< "Fun. Calls count = " */<< modelBasedMethod.getFunctionCallsCount() << "\t";
        std::cout /*<< "Fun. Calls Needed = " */<< modelBasedMethod.getFunctionCallsNeeded() << "\t \t";

        std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(dim, xRosenbrock) << "\t";
        std::cout /*<< "Found value = " */<< resultRosenbrock << "\t";
        std::cout /*<< "Iterations = " */<< rosenbrockMethod.getIterationsCount() << "\t";
        std::cout /*<< "Fun. Calls count = " */<< rosenbrockMethod.getFunctionCallsCount() << "\t";
        std::cout /*<< "Fun. Calls Needed = " */<< rosenbrockMethod.getFunctionCallsNeeded() << "\t";
        std::cout /*<< "Best method = " */<< (modelBasedMethod.getFunctionCallsCount() < rosenbrockMethod.getFunctionCallsCount() ? "MB" : "Ros") << "\t" << "\n";
    }
    return true;
}

void testBench(std::shared_ptr<BM> bm, double eps, double l, double r, bool isFirstBenchmark) {
    
    const int dim = bm->getDim();
    double a[dim], b[dim];
    double x[dim];
    for (int i = 0; i < dim; i++) {
        a[i] = bm->getBounds()[i].first;
        b[i] = bm->getBounds()[i].second;
        x[i] = (b[i] + a[i]) / 2.0;
    }

    bool isFirstLaunch = isFirstBenchmark;

    int steps = 1;
    double stepX = (bm->getBounds()[0].second - bm->getBounds()[0].first) / (steps + 1);
    double stepY = (bm->getBounds()[1].second - bm->getBounds()[1].first) / (steps + 1);
    //std::cout << bm->getBounds()[0].first << ":" << bm->getBounds()[0].second << " " << stepX << "\n";
    //std::cout << bm->getBounds()[1].first << ":" << bm->getBounds()[1].second << " " << stepY << "\n";
    for (int i = 1; i <= steps; i++) {
        for (int j = 1; j <= steps; j++) {
            x[0] = bm->getBounds()[0].first + stepX * i;
            x[1] = bm->getBounds()[1].first + stepY * j;
            //std::cout << snowgoose::VecUtils::vecPrint(dim, x) << " \n";
            testWithInitialPoint(bm, eps, x, a, b, isFirstLaunch);
            isFirstLaunch = false;
        }
    }

    /*x[0] = l;
    x[1] = r;
    testWithInitialPoint(bm, eps, x, a, b, isFirstLaunch);*/
}

int main(int argc, char** argv) {
    //const int dim = argc > 1 ? atoi(argv[1]) : 2;
    const double eps = argc > 3 ? atof(argv[3]) : 1e-4;

    const double x = argc > 1 ? atof(argv[1]) : 0;
    const double y = argc > 2 ? atof(argv[2]) : 0;

    bool isFirstBenchmark = true;

    std::cout << std::fixed << std::setprecision(12);
    /*auto bm = std::make_shared<RosenbrockBenchmark<double>>(dim);
    testBench(bm, eps);*/

    /*auto bm = std::make_shared<GoldsteinPriceBenchmark<double>>();
    testBench(bm, eps);*/

    /*auto bm = std::make_shared<BirdBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<BartelsConnBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<Bohachevsky1Benchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    Benchmarks<double> tests;
    for (auto bm : tests) {
        testBench(bm, eps, x, y, isFirstBenchmark);
        isFirstBenchmark = false;
    }
    return 0;
}

