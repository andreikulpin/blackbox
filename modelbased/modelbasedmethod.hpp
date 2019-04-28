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
            FT mDec = 0.25;
            /**
             * Model adequacy threshold
             */
            FT mRatioThreshold = 0.25;
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

            g = new FT[q - 1];
            G = new FT[n * n];

            FT cauchyPoint[n];
            FT trialPoint[n];

            std::ofstream fout;
            if (mOptions.mDoSavingPath) {
                fout.open("path.log");

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
            
            bool isStopped = false;
            iteration = 0;
            do {
                iteration += 1;
                if (mOptions.mDoTracing) {
                    std::cout << std::endl << " Iteration " << iteration << std::endl;
                    std::cout << std::endl << "Trust region radius = " << trustRegionRadius << std::endl;
                }

                if (mOptions.mDoTracing) {
                    printMatrix("Interpolation set", q, n, Y);
                    printArray("Function values", q, fValues);
                }

                resolveModel();
                resolveCauchyPoint(cauchyPoint);
                
                FT * xCur = Y + xMinIndex * n;
                snowgoose::VecUtils::vecSaxpy(n, xCur, cauchyPoint, 1.0, trialPoint);
                
                FT trialValue = f(trialPoint);
                functionCallsCount++;

                FT trialModelValue = modelValue(fCur, g, G, cauchyPoint);
                FT ratio = (fCur - trialValue) / (fCur - trialModelValue);

                if (mOptions.mDoTracing) {
                    printArray("Cauchy point", n, cauchyPoint);
                    printArray("Trial point", n, trialPoint);
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

                if (ratio >= mOptions.mRatioThreshold) {
                    if (mOptions.mDoSavingPath) {
                        for (int i = 0; i < n; i++) { 
                            fout << xCur[i] << " ";
                        }
                        fout << 0 << std::endl;
                    }

                    if (!isStopped) {
                        const FT dist = snowgoose::VecUtils::vecDist(n, xCur, trialPoint);
                        FT der = (fCur - trialValue) / dist;

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
                        FT maxTrustRegionRadius = calculateMaxTrustRegionRadius(trialPoint, leftBound, rightBound);
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

        FT calculateMaxTrustRegionRadius(FT * x, const FT* leftBound, const FT* rightBound) {
            FT maxTrustRegionRadius = std::min(x[0] - leftBound[0], rightBound[0] - x[0]);
            for (int i = 0; i < n; i++) {
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
                Y[offset + i * n + i] += trustRegionRadius;
            }
            
            offset += n * n;
            for (int i = 0; i < n; i++) {
                Y[offset + i * n + i] += 0.5 * trustRegionRadius;
            }

            offset += n * n;
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - i - 1; j++) {
                    Y[offset + j * n + i] += 0.5 * trustRegionRadius;
                    Y[offset + j * n + i + j + 1] += 0.5 * trustRegionRadius;
                }
                offset += n * (n - i - 1);
            }

            /*if (n > 1) {
                FT rotationMatrix[n];
                std::fill(rotationMatrix, rotationMatrix + n * n, 0.);
                for (int i = 2; i < n; i++) {
                    rotationMatrix[n * i + i] = 1;
                }
                rotationMatrix[0] = cos(alglib::pi() * 0.3);
                rotationMatrix[1] = -sin(alglib::pi() * 0.3);
                rotationMatrix[n] = sin(alglib::pi() * 0.3);
                rotationMatrix[n + 1] = cos(alglib::pi() * 0.3);
                // printMatrix("Rotation matrix", n, n, rotationMatrix);
                
                alglib::real_2d_array matrix;
                matrix.setcontent(n, n, rotationMatrix);
                alglib::real_2d_array vector;
                alglib::real_2d_array result;
                result.setlength(n, 1);

                for (int i = 0; i < q; i++) {
                    vector.setcontent(n, 1, Y + i * n);

                    try {
                        alglib::rmatrixgemm(n, 1, n, 1.0, 
                        matrix, 0, 0, 0, 
                        vector, 0, 0, 0, 
                        0.0, result, 0, 0);
                    } catch (alglib::ap_error e) {
                        std::cout << "ap_error "<< e.msg << std::endl;
                    }

                    for (int j = 0; j < n; j++) {
                        Y[i * n + j] = result[j][0];
                    }
                }
            }*/
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

            /*FT sDet = alglib::rmatrixdet(sMatrix, q - 1);
            std::cout << "S det = " << sDet << std::endl; */

            alglib::real_1d_array fkArray;
            fkArray.setcontent(q - 1, fk);
            alglib::real_1d_array gArray;
            
            alglib::ae_int_t info;
            alglib::densesolverreport rep;
            alglib::rmatrixsolve(sMatrix, q - 1, fkArray, info, rep, gArray);

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

        FT modelValue(FT c, FT * g, FT * G, FT * p) {
            FT gp = snowgoose::VecUtils::vecScalarMult(n, g, p);
            FT pGp = vMvMul(n, p, G);
            //std::cout << "modelValue: gp = " << gp << " pGp = " << pGp << std::endl;
            return c + gp + 0.5 * pGp;
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

            /*std::cout << " vM = " << std::endl;
            AL::printMatrix(vM);

            std::cout << " vMv = " << std::endl;
            AL::printMatrix(vMv);*/

            return vMv(0, 0);
        }

        void printMatrix(const char * name, int n, int m, FT * matrix) {
            std::cout << name << " =\n";
            for (int i = 0; i < n; i++) {
                std::cout << snowgoose::VecUtils::vecPrint(m, &(matrix[i * m])) << "\n";
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

