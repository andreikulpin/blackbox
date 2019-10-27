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

int testMB(std::shared_ptr<BM> bm, double eps, double * xModelBased, double * a, double * b, 
        double * xRos, int takenRosCalls, int overallRosCalls, bool isInitPoint) {
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

    std::function<double (const double*) > func = [&] (const double * xModelBased) {
        return mpp->mObjectives.at(0)->func(xModelBased);
    };

    double resultModelBased = modelBasedMethod.search(dim, xModelBased, a, b, func);

    // bool isResultsClose = snowgoose::VecUtils::vecDistAbs(dim, xModelBased, xRos) < 1e-2;

    // int sumCalls = modelBasedMethod.getFunctionCallsCount() + takenRosCalls;

    // if (isResultsClose || isInitPoint) {
    //     //std::cout /* << "At " */<< snowgoose::VecUtils::vecPrint(dim, xModelBased) << "\t";
    //     //std::cout /*<< "Found value = " */<< resultModelBased << "\t";
    //     //std::cout /*<< "Iterations = " */<< modelBasedMethod.getIterationsCount() << "\t";
    //     std::cout /*<< "Fun. Calls count = " */<< modelBasedMethod.getFunctionCallsCount() << "\t";
    //     //std::cout /*<< "Fun. Calls Needed = " */<< modelBasedMethod.getFunctionCallsNeeded() << "\t \t";
    //     //std::cout /*<< "Ros Calls  taken = " */<< takenRosCalls << "\t";
    //     std::cout /*<< "Ros Overall Calls = " */<< overallRosCalls << "\t";
    //     std::cout /*<< "Sum Calls = " */<< sumCalls << "\t";
    //     //std::cout /*<< "Results close = " */<< (isResultsClose ? "true" : "false") << "\n";

    //     if (isInitPoint) {
    //         std::cout /* << "At " */ << " \t" << snowgoose::VecUtils::vecPrint(dim, xModelBased) << "\t";
    //         std::cout << snowgoose::VecUtils::vecPrint(dim, xRos) << "\t";
    //     }
    //     std::cout << "\n";
    // }
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

    //std::ifstream infile("pathr.log");

    //std::cout << "\nMB started from\t" << "MB Stopped at\t" << "Found value\t" << "Iterations\t" << "Calls count\t" << "Calls needed\t \t" << "Ros calls taken\t" << "Ros overall calls\t"<< "Sum calls\n";  
    //std::cout << "\nMB init X\t" << "MB calls\t" << "Ros calls\t"<< "Hybrid calls\t" << " \t" << "MB stopped at\t" << "Ros stopped at\n";  

    int takenRosCalls;
    double xModelBased[dim];
    double z;
    bool isFirstPoint = true;

    int mbCalls = -1;
    int minCallsCount = rosCalls;
    int minIndex = rosCallCounts.size() - 1;
	/*while (infile >> xInit[0] >> xInit[1] >> z >> takenRosCalls) {
        //std::cout << "\nz = " << z << " takenRosCalls = " << takenRosCalls << "\n";
        testMB(bm, eps, xInit, a, b, xRos, takenRosCalls, rosCalls, isFirstPoint);
        isFirstPoint = false;
	} */

    //snowgoose::VecUtils::vecCopy(dim, xInit, xModelBased);
    //int hybridMbCalls = testMB(bm, eps, xModelBased, a, b, xRos, rosCallCounts[i], rosCalls, isFirstPoint);

    for (int i = 0; i < rosCallCounts.size(); i++) {
        bool isLastPoint = i == rosCallCounts.size() - 1;
        snowgoose::VecUtils::vecCopy(dim, &(rosPath[dim * i]), xModelBased);
        //std::cout << "\nMB init x = " << snowgoose::VecUtils::vecPrint(dim, xModelBased) << " \n";

        int hybridMbCalls;
        if (!isLastPoint) {
            hybridMbCalls = testMB(bm, eps, xModelBased, a, b, xRos, rosCallCounts[i], rosCalls, isFirstPoint);
        } else {
            hybridMbCalls = 0;
        }
        mbCallCounts.push_back(hybridMbCalls);

        bool isResultsClose = snowgoose::VecUtils::vecDistAbs(dim, xModelBased, xRos) < 1e-2;

        if (isResultsClose && i == 0) {
            mbCalls = hybridMbCalls;
        }

        int sumCalls = hybridMbCalls + rosCallCounts[i];

        if (isResultsClose && sumCalls < minCallsCount) {
            minCallsCount = sumCalls;
            minIndex = i;
        }

        /*if (isResultsClose || isFirstPoint || isLastPoint) {
            std::cout << snowgoose::VecUtils::vecPrint(dim, &(rosPath[dim * i])) << "\t";
            std::cout << rosCallCounts[i] << "\t";
            std::cout << hybridMbCalls << "\t";
            std::cout << sumCalls << "\t";

            if (isFirstPoint) {
                std::cout << " \t" << snowgoose::VecUtils::vecPrint(dim, xModelBased) << "\t";
                std::cout << snowgoose::VecUtils::vecPrint(dim, xRos) << "\t";
            }
            std::cout << "\n";
        }*/

        isFirstPoint = false;
    }

    //std::cout << " Min calls count = " << minCallsCount << " index = " << minIndex << " \n"; 

    /*if (isFirstInitPoint) {
        std::cout << "Benchmark\t" << "Glob min x\t" << "Glob min\n";  
        std::cout << bm->getDesc() << "\t";
        std::cout << snowgoose::VecUtils::vecPrint(dim, bm->getGlobMinX().data()) << "\t";
        std::cout << bm->getGlobMinY() << "\n";

        std::cout << "Init X\t" << "Ros calls\t" << "MB calls\t" << "Hybrid calls\t" << "Hybrid Ros calls\t" << "Hybrid MB calls\t" << "% of Ros calls\n";
    }  */

    if (mbCalls > 0) {
        std::cout << bm->getDesc() << "\t";
        std::cout << snowgoose::VecUtils::vecPrint(dim, xInit) << "\t";
        std::cout << rosCalls << "\t";
        std::cout << mbCalls << "\t";
        std::cout << minCallsCount << "\t";
        std::cout << rosCallCounts[minIndex] << "\t";
        std::cout << mbCallCounts[minIndex] << "\t";
        //std::cout << (int)(((double) minCallsCount / rosCalls) * 100) << "\t";
        std::cout << minIndex << "\t";
        std::cout << "\n";
    }

    rosPath.clear();
    rosCallCounts.clear();
    mbCallCounts.clear();

    //infile.close();
    return mbCalls > 0 && minCallsCount < rosCalls;
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

    if (isFirstBenchmark) {
        std::cout << "Benchmark\t" << "Init X\t" << "Ros calls\t" << "MB calls\t" << "Hybrid calls\t" << "Hybrid Ros calls\t" << "Hybrid MB calls\t" << "Min index\n";
    }

    int steps = 2;
    double stepX = (bm->getBounds()[0].second - bm->getBounds()[0].first) / (steps + 1);
    double stepY = (bm->getBounds()[1].second - bm->getBounds()[1].first) / (steps + 1);
    //std::cout << bm->getBounds()[0].first << ":" << bm->getBounds()[0].second << " " << stepX << "\n";
    //std::cout << bm->getBounds()[1].first << ":" << bm->getBounds()[1].second << " " << stepY << "\n";
    for (int i = 1; i <= steps; i++) {
        for (int j = 1; j <= steps; j++) {
            xInit[0] = bm->getBounds()[0].first + stepX * i;
            xInit[1] = bm->getBounds()[1].first + stepY * j;
            //std::cout << snowgoose::VecUtils::vecPrint(dim, xInit) << " \n";
            isFirstLaunch = testWithInitialPoint(bm, eps, xInit[0], xInit[1], isFirstLaunch);
            isFirstLaunch = false;
        }
    }

    /*xInit[0] = x;
    xInitx[1] = y;
    testWithInitialPoint(bm, eps, xInit[0], xInit[1], isFirstLaunch);*/

    //std::cout << "\n\n\n";
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

    /*auto bm = std::make_shared<BirdBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<BartelsConnBenchmark<double>>();
    testBench(bm, eps, x, y, true);*/

    auto bm = std::make_shared<Price2Benchmark<double>>();
    testBench(bm, eps, x, y, true);

    /*bm = std::make_shared<BealeBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

    /*auto bm = std::make_shared<Alpine1Benchmark<double>>(3); 
    testBench(bm, eps, x, y, true);

    auto bm2 = std::make_shared<Alpine2Benchmark<double>>(); 
    testBench(bm2, eps, x, y, true);*/

    /*bm = std::make_shared<GoldsteinPriceBenchmark<double>>();
    testBench(bm, eps, x, y, false);*/

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

