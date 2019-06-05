/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   modelbasedmethod.hpp
 * Author: andrei
 *
 * Created on January 20, 2019, 5:41 PM
 */

#ifndef MODELBASEDMETHOD_HPP
#define MODELBASEDMETHOD_HPP

#include <common/bbsolver.hpp>
#include <common/vec.hpp>
#include <cmath>
#include <fstream>
#include "ap.h"
#include "linalg.h"
#include "solvers.h"
#include "alglibutils.hpp"

namespace LOCSEARCH {

    template <typename FT> class ModelBasedMethod : public BlackBoxSolver<FT> {
    public:

        /**
         * Determines stopping conditions
         * @param fval current best value found
         * @param x current best point
         * @param stpn step number
         * @return true if the search should stop
         */
        using Stopper = std::function<bool(FT fval, const FT* x, int stpn) >;

        /**
         * Watches the current step
         * @param fval current best value found
         * @param current best point
         * @param gran current granularity vector
         * @param grad current gradient estimate
         * @param success true if the coordinate descent was successful
         * @param dirs ortogonalized directions
         * @param stpn stage number
         */
        using Watcher = std::function<void(FT fval, const FT* x, const std::vector<FT>& gran, bool success, FT grad,  FT* dirs, int stpn) >;

        /**
         * Options for Model Based method
         */
        struct Options {
            /**
             * Initial trust region radius
             */
            FT mInitialRadius = 1.;
            /**
             * Increase in the case of success
             */
            FT mInc = 2;
            /**
             * Decrease in the case of failure
             */
            FT mDec = 0.5;
            /**
             * Model adequacy threshold
             */
            FT mRatioThreshold = 0.25;
            /**
             * Minimal trust region radius
             */
            FT mMinTrustRegionRadius = 1e-06;
            /**
             * Minimal gradient estimation to stop
             */
            FT mMinGrad = 1e-03;
            /**
             * Function value approximation
             */
            FT mEps = 1e-03;
            /**
             * Total max steps number
             */
            int mMaxIterations = 200;
            /**
             * Known function global min
             */
            FT mFunctionGlobMin;
            /**
             * Trace on/off
             */
            bool mDoTracing = false;
            /**
             * Save search path on/off
             */
            bool mDoSavingPath = false;
            /**
             * Use Dogleg method to solve subproblem when its possible
             */
            bool useDogleg = true;
            /**
             * File name to save path
             */
            const char * pathFile = "path.log";
        };

        /**
         * The constructor
         */
        ModelBasedMethod() {}

        /**
         * Performs search
         * @param x start point and result
         * @return found value
         */
        FT search(int n, FT* x, const FT* leftBound, const FT* rightBound, const std::function<FT ( const FT* )> &f) override {
            this->n = n;
            q = (n + 1) * (n + 2) * 0.5;

            FT maxTrustRegionRadius = calculateMaxTrustRegionRadius(x, leftBound, rightBound);
            trustRegionRadius = std::min(mOptions.mInitialRadius, maxTrustRegionRadius);

            Y = new FT[q * n];
            fValues = new FT[q];

            g = new FT[n];
            G = new FT[n * n];

            FT cauchyPoint[n];
            FT trialPoint[n];

            std::ofstream fout;
            if (mOptions.mDoSavingPath) {
                fout.open(mOptions.pathFile);

                for (int i = 0; i < n; i++) { 
                    fout << x[i] << " ";
                }
                fout << 0 << std::endl;
            }

            functionCallsCount = 0;
            fillInterpolationSet(x);  

            xMinIndex = 0;
            fValues[xMinIndex] = f(x);
            functionCallsCount++;
            evaluateInterpolationSet(f); 

            maxTrustRegionRadius = calculateMaxTrustRegionRadius(Y + xMinIndex * n, leftBound, rightBound);
            trustRegionRadius = std::min(trustRegionRadius, maxTrustRegionRadius);
            
            bool isStopped = false;
            iteration = 0;
            do {
                iteration += 1;
                if (mOptions.mDoTracing) {
                    std::cout << std::endl << " Iteration " << iteration << std::endl;
                    std::cout << std::endl << "Trust region radius = " << trustRegionRadius << std::endl;
                }

                if (mOptions.mDoTracing) {
                    //printMatrix("Interpolation set", q, n, Y);
                    printArray("Function values", q, fValues);
                }

                resolveModel();

                bool isGessianPositiveDefinite = isMatrixPositiveDefinite(n, G);
                //std::cout << "isGessianPositiveDefinite = " << isGessianPositiveDefinite << "\n"; 

                if (isGessianPositiveDefinite && mOptions.useDogleg) {
                    bool isSuccess = dogleg(cauchyPoint);
                    //std::cout << "Dogleg " << (isSuccess ? "success" : "not success") << "\n";
                    if (!isSuccess) {
                        resolveCauchyPoint(cauchyPoint);
                    }
                } else {
                    resolveCauchyPoint(cauchyPoint);
                }

                FT * xCur = Y + xMinIndex * n;
                snowgoose::VecUtils::vecSaxpy(n, xCur, cauchyPoint, 1.0, trialPoint);
                
                FT trialValue = f(trialPoint);
                functionCallsCount++;

                FT trialModelValue = modelValue(fCur, g, G, cauchyPoint);
                FT ratio = (fCur - trialValue) / (fCur - trialModelValue);

                if (mOptions.mDoTracing) {
                    printArray("Current point", n, xCur);
                    printArray("Trial point", n, trialPoint);
                    std::cout << "Current function value = " << fCur << std::endl;
                    std::cout << "Trial function value = " << trialValue << std::endl;
                    std::cout << "Trial function value = " << trialValue << std::endl;
                    std::cout << "Trial model value = " << trialModelValue << std::endl;
                    std::cout << "Ratio = " << ratio << std::endl;
                }

                if (iteration >= mOptions.mMaxIterations) {
                    if (mOptions.mDoTracing) 
                        std::cout << "Stopped as steps number was too big" << std::endl;
                    isStopped = true;
                }

                if (std::abs(fCur - mOptions.mFunctionGlobMin) < mOptions.mEps) {
                    if (mOptions.mDoTracing) 
                        std::cout << "Stopped as estimation was accurate" << std::endl;
                    isStopped = true;
                }

                if (trustRegionRadius < mOptions.mMinTrustRegionRadius) {
                    if (mOptions.mDoTracing) 
                        std::cout << "Stopped as trust region radius was less than " << mOptions.mMinTrustRegionRadius << std::endl;
                    isStopped = true;
                }

                const FT dist = snowgoose::VecUtils::vecDist(n, xCur, trialPoint);
                FT der = (fCur - trialValue) / dist;
                if (mOptions.mDoSavingPath) {
                    for (int i = 0; i < n; i++) { 
                        fout << xCur[i] << " ";
                    }
                    fout << 0 << std::endl;
                }
                    
                if (ratio >= mOptions.mRatioThreshold) {
                    if (!isStopped) {
                        const FT dist = snowgoose::VecUtils::vecDist(n, xCur, trialPoint);
                        FT der = (fCur - trialValue) / dist;

                        if (mOptions.mDoTracing)
                            std::cout << "Grad = " << der << "\n";

                        if (der < mOptions.mMinGrad) {
                            if (mOptions.mDoTracing) 
                                std::cout << "Stopped as gradient estimate was less than " << mOptions.mMinGrad << std::endl;
                            isStopped = true;
                        } else {
                            xMinIndex = xMaxIndex;
                            snowgoose::VecUtils::vecCopy(n, trialPoint, Y + xMinIndex * n);
                            fValues[xMinIndex] = trialValue;
                            findMaxValue();
                        }
                    }
                    
                    fCur = trialValue;

                    if (!isStopped) {
                        maxTrustRegionRadius = calculateMaxTrustRegionRadius(trialPoint, leftBound, rightBound);
                        if (maxTrustRegionRadius == 0) {
                            if (mOptions.mDoTracing) 
                                std::cout << "Stopped as bound was reached" << std::endl;
                            isStopped = true;
                        } else {
                            trustRegionRadius = std::min(trustRegionRadius * mOptions.mInc, maxTrustRegionRadius);
                        }
                    }
                    
                } else if (!isStopped) {
                    trustRegionRadius *= mOptions.mDec;
                    fillInterpolationSet(xCur);
                    evaluateInterpolationSet(f);
                    maxTrustRegionRadius = calculateMaxTrustRegionRadius(Y + xMinIndex * n, leftBound, rightBound);
                    trustRegionRadius = std::min(trustRegionRadius, maxTrustRegionRadius);
                }

                if (mOptions.mDoTracing) {
                    std::cout << "Function calls count = " << functionCallsCount << std::endl;
                }

            } while (!isStopped);

            snowgoose::VecUtils::vecCopy(n, Y + xMinIndex * n, x);

            fout.close();
            delete [] Y;
            delete [] fValues;
            delete [] g;
            delete [] G;
            return fCur;
        }

        std::string about() const {
            std::ostringstream os;
            os << "Model Based method\n";
            os << "options:\n";
            os << "initial trust region radius = " << mOptions.mInitialRadius << "\n";
            os << "decrement = " << mOptions.mDec << "\n";
            os << "increment = " << mOptions.mInc << "\n";
            os << "ratio threshold = " << mOptions.mRatioThreshold << "\n";
            os << "accuracy = " << mOptions.mEps << "\n";
            os << "lower bound on gradient = " << mOptions.mMinGrad << "\n";
            os << "maxima iterations = " << mOptions.mMaxIterations << "\n";
            return os.str();
        }

        /**
         * Retrieve options
         * @return options
         */
        Options & getOptions() {
            return mOptions;
        }

        /**
         * Retrieve stoppers vector reference
         * @return stoppers vector reference
         */
        std::vector<Stopper>& getStoppers() {
            return mStoppers;
        }

        /**
         * Get watchers' vector
         * @return watchers vector
         */
        std::vector<Watcher>& getWatchers() {
            return mWatchers;
        }

        int getFunctionCallsCount() {
            return functionCallsCount;
        }

        int getIterationsCount() {
            return iteration;
        }


    private:
        Options mOptions;
        std::vector<Stopper> mStoppers;
        std::vector<Watcher> mWatchers;

        int n;
        int q;

        double trustRegionRadius;
        FT * Y; // Interpolation set
        FT * fValues;
        FT fCur;
        int xMinIndex;
        FT fMax;
        int xMaxIndex;

        FT * g;
        FT * G;

        int iteration;
        int functionCallsCount;

        bool isMatrixPositiveDefinite(int n, FT * matrix) {
            alglib::real_2d_array alMatrix;
            alMatrix.setcontent(n, n, matrix);

            for (int i = 1; i <= n; i++) {
                FT det = alglib::rmatrixdet(alMatrix, i);

                //std::cout << "det = " << det << "\n";

                if (det < 0) return false;
            }

            return true;
        }

        FT calculateMaxTrustRegionRadius(FT * x, const FT* leftBound, const FT* rightBound) {
            /*std::cout << snowgoose::VecUtils::vecPrint(n, x) << " "
            << snowgoose::VecUtils::vecPrint(n, leftBound) << " "
            << snowgoose::VecUtils::vecPrint(n, rightBound) << "\n";*/
            FT maxTrustRegionRadius = std::min(x[0] - leftBound[0], rightBound[0] - x[0]);
            for (int i = 1; i < n; i++) {
                FT minDistance = std::min(x[i] - leftBound[i], rightBound[i] - x[i]);
                if (minDistance < maxTrustRegionRadius) {
                    maxTrustRegionRadius = minDistance;
                }
            }

            return maxTrustRegionRadius;
        }

        void fillInterpolationSet(FT * x) {
            for (int i = 0; i < q; i++) {
                for (int j = 0; j < n; j++) {
                    Y[i * n + j] = x[j];
                }
            }
           
            int offset = n;
            for (int i = 0; i < n; i++) {
                Y[offset + i * n + i] += 0.5 * trustRegionRadius;
            }
            
            offset += n * n;
            for (int i = 0; i < n; i++) {
                Y[offset + i * n + i] -= 0.25 * trustRegionRadius;
            }

            offset += n * n;
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - i - 1; j++) {
                    Y[offset + j * n + i] += 0.25 * trustRegionRadius;
                    Y[offset + j * n + i + j + 1] += 0.25 * trustRegionRadius;
                }
                offset += n * (n - i - 1);
            }
        }

        void evaluateInterpolationSet(const std::function<FT ( const FT* )> &f) {
            FT yi[n];

            //std::cout << snowgoose::VecUtils::vecPrint(n, xcur) << " " << fcur << std::endl;
            for (int i = 0; i < q; i++) {
                if (i != xMinIndex) {
                    snowgoose::VecUtils::vecCopy(n, Y + i * n, yi);
                    fValues[i] = f(yi);
                    functionCallsCount++;
                }
                // std::cout << snowgoose::VecUtils::vecPrint(n, yi) << " " << value << std::endl;
            }

            findMinValue();
            findMaxValue();

            //std::cout << "xcur " << xMinIndex << " " << fCur << std::endl;
            //std::cout << "xMax " << xMaxIndex << " " << fMax << std::endl;
        }

        void findMinValue() {
            xMinIndex = 0;
            fCur = fValues[0];
            for (int i = 1; i < q; i++) {
                if (fValues[i] < fCur) {
                    fCur = fValues[i];
                    xMinIndex = i;
                }
            }
        }

        void findMaxValue() {
            xMaxIndex = 0;
            fMax = fValues[0];
            for (int i = 1; i < q; i++) {
                if (fValues[i] > fMax) {
                    fMax = fValues[i];
                    xMaxIndex = i;
                }
            }
        }

        void resolveModel() {
            //std::cout << "xMinIndex = " << xMinIndex << std::endl;
            FT S[(q - 1) * (q - 1)];
            std::fill(S, S + (q - 1) * (q - 1), 0.);
            
            FT sl[n];
            FT fk[q - 1];
            
            FT * xCur = Y + xMinIndex * n;
            
            for (int yIndex = 0; yIndex < q; yIndex++) {
                if (yIndex == xMinIndex) continue;
                
                snowgoose::VecUtils::vecCopy(n, Y + yIndex * n, sl);
                alglib::vsub(sl, xCur, n);
                //printArray("sl", n, sl);
                
                int sIndex = (yIndex < xMinIndex) ? yIndex : yIndex - 1;
                //std::cout << " i = " << i << " j = " << j << std::endl;
                snowgoose::VecUtils::vecCopy(n, sl, S + sIndex * (q - 1));
                
                int offset = sIndex * (q - 1) + n;
                for (int i = 0; i < n - 1; i++) {
                    for (int j = i + 1; j < n; j++) {
                        //std::cout << "Index " << offset + j - i - 1 << std::endl;
                        S[offset + j - i - 1] = sl[i] * sl[j];
                    }
                    offset += (n - i - 1);
                }
                
                for (int i = 0; i < n; i++) {
                    S[offset + i] = sl[i] * sl[i] / sqrt(2);
                }
                
                fk[sIndex] = fValues[yIndex] - fCur;
            }
            
            //printMatrix("S", (q - 1), (q - 1), S);
            //printArray("fk", q - 1, fk);
            
            alglib::real_2d_array sMatrix;
            sMatrix.setcontent(q - 1, q - 1, S);
            alglib::real_1d_array fkArray;
            fkArray.setcontent(q - 1, fk);
            
            alglib::real_1d_array gArray;
            gArray.setlength(q-1);
            
            alglib::ae_int_t info;
            alglib::densesolverreport rep;
            //std::cout << "test\n";
            alglib::rmatrixsolve(sMatrix, q - 1, fkArray, info, rep, gArray);
            //std::cout << "Solving info = " << info << "\n";

            //printArray("gk", q - 1, &gArray[0]);

            int offset = 0;
            for (int i = 0; i < n - 1; i++) {
                offset += n - i;
                for (int j = 0; j < n - i - 1; j++) {
                    //std::cout << " " << i * n + i + 1 + j << " " << offset + j << std::endl;
                    G[i * n + i + 1 + j] = G[(i + j + 1) * n + i] = gArray[offset + j];
                }
            }

            offset += 1;
            //std::cout << " offset = " << offset << std::endl;
            for (int i = 0; i < n; i++) {
                G[i * n + i] = gArray[offset + i] * sqrt(2);
                g[i] = gArray[i];
            }
            //printArray("g", n, g);
            //printMatrix("G", n, n, G);
        }

        void resolveCauchyPoint(FT * cauchyPoint) {
            FT gGg = vMvMul(n, g, G);
            //std::cout << " gGg = " << gGg << std::endl;

            FT gNorm = snowgoose::VecUtils::vecNormTwo(n, g);
            //std::cout << std::endl << " ||g|| = " << gNorm << std::endl;

            FT tau;
            if (gGg <= 0) {
                tau = 1.;
            } else {
                FT minimizer = pow(gNorm, 3.) / (trustRegionRadius * gGg);
                tau = std::min(minimizer, 1.);
            }

            //std::cout << std::endl << " tau = " << tau << std::endl;

            FT alpha = - tau * trustRegionRadius / gNorm;
            //std::cout << std::endl << " alpha = " << alpha << std::endl;
            snowgoose::VecUtils::vecMult(n, g, alpha, cauchyPoint);
        }

        bool dogleg(FT * stepPoint) {
            FT fullStep[n];
            resolveFullStepPoint(g, G, fullStep);
            //printArray("Full step", n, fullStep);
            FT fullStepNorm = snowgoose::VecUtils::vecNormTwo(n, fullStep);
            //std::cout << "fullStepNorm = " << fullStepNorm << "\n"; 

            if (fullStepNorm <= trustRegionRadius) {
                snowgoose::VecUtils::vecCopy(n, fullStep, stepPoint);
                return true;
            }

            if (trustRegionRadius < 0.001) {
                FT gNorm = snowgoose::VecUtils::vecNormTwo(n, g);
                snowgoose::VecUtils::vecMult(n, g, (- trustRegionRadius / gNorm), stepPoint);
                return true;
            }

            FT pU[n];
            FT gg = snowgoose::VecUtils::vecNormTwoSqr(n, g);
            FT gGg = vMvMul(n, g, G);
            snowgoose::VecUtils::vecMult(n, g, (-gg / gGg), pU);
            FT pUNorm = snowgoose::VecUtils::vecNormTwo(n, pU);
            //printArray("pU", n, pU);
            //std::cout << "pUNorm = " << pUNorm << "\n"; 

            FT tau = (trustRegionRadius - pUNorm) / (fullStepNorm - pUNorm) + 1;
            //std::cout << "tau = " << tau << "\n"; 

            if (tau >= 0 && tau <= 1) {
                snowgoose::VecUtils::vecMult(n, pU, tau, stepPoint);
                return true;
            }

            if (tau >= 1 && tau <= 2) {
                FT sub[n];
                snowgoose::VecUtils::vecSaxpy(n, fullStep, pU, -1.0, sub);
                snowgoose::VecUtils::vecSaxpy(n, pU, sub, (tau - 1), stepPoint);
                return true;
            }

            return false;
        }

        void resolveFullStepPoint(FT * g, FT * G, FT * fullStep) {
            alglib::real_2d_array gVector;
            alglib::real_2d_array gMatrix;
            alglib::real_2d_array gInvMatrix;

            alglib::matinvreport matinvrep; 
            alglib::ae_int_t info;

            gMatrix.setcontent(n, n, G);
            //printMatrix("G", gMatrix);

            alglib::spdmatrixinverse(gMatrix, info, matinvrep); 
            //printMatrix("G inverted", gMatrix);
            
            FT GInv[n * n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    GInv[i * n + j] = gMatrix(i, j);
                }
            }

            MvMul(n, GInv, g, fullStep);
            //printArray("Gg", n, fullStep);
            snowgoose::VecUtils::revert(n, fullStep);
            //printArray("Gg", n, fullStep);
            
            //alglib::real_2d_array fullStepPoint;
        }

        FT modelValue(FT c, FT * g, FT * G, FT * p) {
            FT gp = snowgoose::VecUtils::vecScalarMult(n, g, p);
            FT pGp = vMvMul(n, p, G);
            //std::cout << "modelValue: gp = " << gp << " pGp = " << pGp << std::endl;
            return c + gp + 0.5 * pGp;
        }

        void MvMul(const int n, FT * M, FT * v, FT * Mv) {
            alglib::real_2d_array vector;
            alglib::real_2d_array matrix;
            alglib::real_2d_array MvMatrix;

            vector.setcontent(n, 1, v);
            matrix.setcontent(n, n, M);
            MvMatrix.setlength(n, 1);

            try {
                alglib::rmatrixgemm(n, 1, n, 1.0, 
                matrix, 0, 0, 0, 
                vector, 0, 0, 0, 
                0.0, MvMatrix, 0, 0);
            } catch (alglib::ap_error e) {
                std::cout << "ap_error "<< e.msg << std::endl;
            }

            for (int i = 0; i < n; i++) {
                Mv[i] = MvMatrix(i, 0);
            }
        }

        FT vMvMul(const int n, FT * v, FT * M) {
            alglib::real_2d_array vector;
            alglib::real_2d_array matrix;
            alglib::real_2d_array vM;
            alglib::real_2d_array vMv;

            vector.setcontent(1, n, v);
            matrix.setcontent(n, n, M);
            vM.setlength(1, n);
            vMv.setlength(1, 1);

            try {
                alglib::rmatrixgemm(1, n, n, 1.0, 
                vector, 0, 0, 0, 
                matrix, 0, 0, 0, 
                0.0, vM, 0, 0);

                alglib::rmatrixgemm(1, 1, n, 1.0, 
                vM, 0, 0, 0, 
                vector, 0, 0, 1, 
                0.0, vMv, 0, 0);
            } catch (alglib::ap_error e) {
                std::cout << "ap_error "<< e.msg << std::endl;
            }

            return vMv(0, 0);
        }

        void printMatrix(const char * name, int n, int m, FT * matrix) {
            std::cout << name << " =\n";
            for (int i = 0; i < n; i++) {
                std::cout << snowgoose::VecUtils::vecPrint(m, &(matrix[i * m])) << "\n";
            }
        }

        void printMatrix(const char * name, alglib::real_2d_array matrix) {
            std::cout << name << " =\n";
            for (int i = 0; i < matrix.rows(); i++) {
                for (int j = 0; j < matrix.cols(); j++) {
                    std::cout << matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }

        void printArray(const char * name, int n, FT * array) {
            std::cout << name << " = ";
            std::cout << snowgoose::VecUtils::vecPrint(n, array) << "\n";
        }

        void printVector(const char * name, int n, std::vector<FT> vector) {
            std::cout << name << " = ";
            std::cout << "[ ";
            for (int i = 0; i < n; i++) {
                std::cout << vector[i] << ", ";
            }
            std::cout << " ]" << "\n";
        }
    };
}

#endif /* MODELBASEDMETHOD_HPP */

