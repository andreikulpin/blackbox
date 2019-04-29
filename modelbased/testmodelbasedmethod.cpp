/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   testmodelbasedmethod.cpp
 * Author: andrei
 *
 * Created on January 20, 2019, 5:44 PM
 */


#include <iostream>
#include "modelbasedmethod.hpp"
#include "math.h"

using namespace std;

// Rosenbrock function
double func(const double* x) {
    return 100 * SGSQR(x[1] - x[0] * x[0]) + SGSQR(1 - x[1]);
}

// Adjiman function
double func2(const double* x) {
    return cos(x[0]) * sin(x[1]) - x[0] / ((x[1] * x[1]) + 1);
}

int main(int argc, char** argv) {
    const int dim = 2;
    double x[dim] = {0, 0};
    double a[dim], b[dim];
    std::fill(a, a + dim, -4);
    std::fill(b, b + dim, 8);

    const int maxIterations = argc > 1 ? atoi(argv[1]) : 100;

    LOCSEARCH::ModelBasedMethod<double> searchMethod;
    searchMethod.getOptions().mMaxIterations = maxIterations;
    searchMethod.getOptions().mFunctionGlobMin = 0.0;
    searchMethod.getOptions().mMinGrad = 1e-3;
    searchMethod.getOptions().mEps = 1e-8;
    searchMethod.getOptions().mDoTracing = true;
    searchMethod.getOptions().mDoSavingPath = true;
    
    double result = searchMethod.search(dim, x, a, b, func);
    
    std::cout << searchMethod.about() << "\n";
    std::cout << "Found value = " << result << "\n";
    return 0;
}

