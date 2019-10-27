/* 
 * File:   rosenbrockmethod.hpp
 * Author: Andrey Kulpin 
 */

#ifndef ROSENBROCKMETHOD_HPP
#define  ROSENBROCKMETHOD_HPP

#include <sstream>
#include <vector>
#include <functional>
#include <memory>
#include <common/bbsolver.hpp>
//#include <common/dummyls.hpp>
#include <common/vec.hpp>
//#include <common/sgerrcheck.hpp>
//#include <mpproblem.hpp>
//#include <mputils.hpp>
//#include <common/lineseach.hpp>
#include <cmath>
#include <fstream>


namespace LOCSEARCH {

    /**
     * Rosenbrock method 
     * Description here: Rosenbrock, H. (1960). An automatic method for finding the greatest or least value of a function. The Computer Journal, 3(3), 175-184.
     */
    template <typename FT> class RosenbrockMethod : public BlackBoxSolver<FT> {
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
         * Options for Rosenbrock method
         */
        struct Options {
            /**
             * Initial values of granularity for each direction
             */
            std::vector<FT> mHInit;
            /**
             * Increase in the case of success
             */
            FT mInc = 2;
            /**
             * Decrease in the case of failure
             */
            FT mDec = 0.5;
            /**
             * Lower bound for granularity
             */
            FT mHLB = 1e-03;
            /**
             * Upper bound on granularity
             */
            FT mHUB = 1e+02;
            /**
             * Mininam gradient extimation to do ortoganalization
             */
            FT mMinGrad = 1e-03;
            /**
             * Do ortoganization
             */
            bool mDoOrt = true;

            /**
             * Total max steps number
             */
            int mMaxStepsNumber = 100;
            /**
             * Trace on/off
             */
            bool mDoTracing = false;
            /**
             * Known function global min
             */
            FT mFunctionGlobMin;
            /**
             * Function value approximation
             */
            FT mEps = 1e-03;
        };

        /**
         * Performs search
         * @param x start point and result
         * @param v  the resulting value
         * @return true if search converged and false otherwise
         */
        FT search(int n, FT* x, const FT* leftBound, const FT* rightBound, const std::function<FT ( const FT* )> &f) override {
            const int nsqr = n * n;
            functionCallsCount = 0;
            functionCallsCountNeeded = 0;

            std::ofstream fout;
            fout.open("pathr.log");

            for (int i = 0; i < n; i++) { 
                fout << x[i] << " ";
                path.push_back(x[i]);
            }
            callCounts.push_back(functionCallsCount);
            if (n < 3)
                fout << 0;
            fout << std::endl;

            double v;
            FT fcur = f(x);
            FT xOld[n];

            functionCallsCount++;
            functionCallsCountNeeded = functionCallsCount;

            std::vector<FT> sft(mOptions.mHInit);
            std::vector<FT> stepLen(n, 0);

            FT * dirs = new FT[nsqr];
            snowgoose::VecUtils::vecSet(n * n, 0., dirs);
            for (int i = 0; i < n; i++) {
                dirs[i * n + i] = 1;
            }

            auto printDirs = [&dirs, n] () {
                std::cout << "==== dirs ====\n";
                for (int i = 0; i < n; i++) {
                    FT* d = dirs + i * n;
                    std::cout << "[";
                    for (int j = 0; j < n; j++) {
                        std::cout << d[j] << " ";
                    }
                    std::cout << "]\n";
                }
                std::cout << "==============\n";
            };

            FT * a = new FT[n];
            FT * b = new FT[nsqr];
            FT * d = new FT[nsqr];

            stageNum = 1;
            bool br = false;

            auto inc = [this] (FT h) {
                FT t = h;
                t = h * mOptions.mInc;
                t = std::min(t, mOptions.mHUB);
                return t;
            };

            auto dec = [this](FT h) {
                FT t = h;
                t = h * mOptions.mDec;
                t = std::max(t, mOptions.mHLB);
                return t;
            };

            /*
             * Attepmt yielding new minimum along each base direction.
             * @return true if step along at least one direction was successful
             */
            auto step = [&] () {
                bool isStepSuccessful = false;
                FT xn[n];
                snowgoose::VecUtils::vecCopy(n, x, xn);

                for (int i = 0; i < n; i++) {
                    const FT h = sft[i];
                    FT xtmp[n];
                    snowgoose::VecUtils::vecSaxpy(n, xn, &(dirs[i * n]), h, xtmp);
                    if (isInBox(n, xtmp, leftBound, rightBound)) {
                        FT ftmp = f(xtmp);
                        functionCallsCount++;

                        if (ftmp < fcur) {
                            isStepSuccessful = true;
                            stepLen[i] += h;
                            sft[i] = inc(h);
                            snowgoose::VecUtils::vecCopy(n, xtmp, xn);
                            fcur = ftmp;
                        } else {
                            const FT nh = dec(std::abs(h));
                            sft[i] = (h > 0) ? - nh : nh;
                        }
                    } else {
                        const FT nh = dec(std::abs(h));
                        sft[i] = (h > 0) ? - nh : nh;
                    }
                }

                snowgoose::VecUtils::vecCopy(n, xn, x);
                return isStepSuccessful;
            };

            auto ortogonalize = [&] () {

                for (int i = 0; i < n; i++) {
                    if (stepLen[i] == 0) {
                        snowgoose::VecUtils::vecCopy(n, &(dirs[i * n]), a);

                    } else {
                        snowgoose::VecUtils::vecSet(n, 0., a);
                        for (int j = i; j < n; j++) {
                            snowgoose::VecUtils::vecSaxpy(n, a, &(dirs[j * n]), stepLen[j], a);
                        }
                    }

                    snowgoose::VecUtils::vecCopy(n, a, &(b[i * n]));

                    for (int j = 0; j < i; j++) {
                        FT scalarMlp = snowgoose::VecUtils::vecScalarMult(n, a, &(d[j * n]));
                        snowgoose::VecUtils::vecSaxpy(n, &(b[i * n]), &(d[j * n]), -scalarMlp, &(b[i * n]));
                    }

                    FT norm = snowgoose::VecUtils::vecNormTwo(n, &(b[i * n]));
                    snowgoose::VecUtils::vecMult(n, &(b[i * n]), 1 / norm, &(d[i * n]));
                }

                for (int i = 0; i < n; i++) {
                    snowgoose::VecUtils::vecCopy(n, &(d[i * n]), &(dirs[i * n]));
                }

            };

            while (!br) {
                FT der;
                FT xold[n];
                snowgoose::VecUtils::vecCopy(n, x, xold);
                const FT fold = fcur;
                const bool success = step();
                if(success) {
                    functionCallsCountNeeded = functionCallsCount;

                    for (int i = 0; i < n; i++) { 
                        fout << x[i] << " ";
                        path.push_back(x[i]);
                    }
                    callCounts.push_back(functionCallsCount);
                    if (n < 3)
                        fout << 0;
                    fout << std::endl;

                    const FT dist = snowgoose::VecUtils::vecDist(n, x, xold);
                    der = (fold - fcur) / dist;
                    if (mOptions.mDoTracing) 
                        std::cout << "Grad Ros = " << der << "\n";
//                    std::cout << "der = " << der << std::endl;
                    if(der < mOptions.mMinGrad) {
                       if (mOptions.mDoTracing) 
                            std::cout << "Stopped as gradient estimate is less than " << mOptions.mMinGrad << std::endl;
                        break;
                    }
                    // HACK
//                    const FT mult = 2. * std::abs(fcur) / der;
//                    for(int i = 0; i < n; i ++) {
//                        sft[i] = (sft[i] > 0 ? 1 : - 1) * mOptions.mHInit[i] * mult;
//                    }
                    // HACK
                }

                if (mOptions.mDoTracing) {
                    std::cout << " Stage " << stageNum << "\n"; 
                    std::cout << "X = " << snowgoose::VecUtils::vecPrint(n, x) << "\n";
                    std::cout << "fCur = " << fcur << "\n";
                    std::cout << "Grad Ros = " << der << "\n";
                    std::cout << "Calls count = " << functionCallsCount << "\n";
                }
                
                stageNum ++;


                if (!success) {
                    br = true;
                    for (int i = 0; i < n; i++) {
                        if (std::abs(sft[i]) > mOptions.mHLB) {
                            br = false;
                            break;
                        }
                    }

                    if (!br) {
                        //std::cout << snowgoose::VecUtils::vecPrint(n, stepLen.data()) << std::endl;
                        if (mOptions.mDoOrt) {
                            ortogonalize();
                            //                                sft = mOptions.mHInit;
                            stepLen.assign(n, 0);
                        }
                    } else if (mOptions.mDoTracing) {
                        std::cout << "Stopped as all step lengths was less than " << mOptions.mHLB << "\n";
                    }
                }

                if (std::abs(fcur - mOptions.mFunctionGlobMin) < mOptions.mEps) {
                    if (mOptions.mDoTracing) 
                        std::cout << "Stopped as estimation was accurate" << std::endl;
                    br = true;
                }

                if (stageNum >= mOptions.mMaxStepsNumber) {
                    if (mOptions.mDoTracing) 
                        std::cout << "Stopped as number of stages was too big\n";
                    br = true;
                }

                for (auto w : mWatchers) {
                    w(fcur, x, sft, success, der, dirs, stageNum);
                }
                for (auto s : mStoppers) {
                    if (s(fcur, x, stageNum)) {
                        br = true;
                        break;
                    }
                }
            }
            v = fcur;

            if (mOptions.mDoTracing) {
                std::cout << "Function calls count = " << functionCallsCount << std::endl;
            }

            fout.close();
            delete [] dirs;
            delete [] a;
            delete [] b;
            delete [] d;
            return v;
        }

        std::string about() const {
            std::ostringstream os;
            os << "Rosenbrock method\n";
            os << "options:\n";
            os << "decrement = " << mOptions.mDec << "\n";
            os << "increment = " << mOptions.mInc << "\n";
            os << "initial step size = [ ";
            for(auto a : mOptions.mHInit) 
                os << " " << a;
            os << " ]\n";
            os << "bounds on step size = [" << mOptions.mHLB << " " << mOptions.mHUB << "]\n";
            os << "lower bound on gradient = " << mOptions.mMinGrad << "\n";
            os << "maxima stages = " << mOptions.mMaxStepsNumber << "\n";
            os << (mOptions.mDoOrt ? "do ortogonalization\n" : "don't do ortogonalization\n");
            os << (mOptions.mDoTracing ? "do tracing\n" : "don't do tracing\n");
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
            return stageNum;
        }

        int getFunctionCallsNeeded() {
            return functionCallsCountNeeded;
        }

        std::vector<double>& getPath() {
            return path;
        }

        std::vector<int>& getCallCounts() {
            return callCounts;
        }

    private:
        Options mOptions;
        std::vector<Stopper> mStoppers;
        std::vector<Watcher> mWatchers;
        int stageNum;
        int functionCallsCount;
        int functionCallsCountNeeded;

        std::vector<double> path;
        std::vector<int> callCounts;

        void printMatrix(const char * name, int n, int m, FT * matrix) {
            std::cout << name << " =\n";
            for (int i = 0; i < n; i++) {
                std::cout << snowgoose::VecUtils::vecPrint(m, &(matrix[i * n])) << "\n";
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
        
        bool isInBox(int n, const FT* x, const FT* a, const FT* b) {
            for (int i = 0; i < n; i++) {
                if (x[i] > b[i]) {
                    return false;
                }
                
                if (x[i] < a[i]) {
                    return false;
                }
            }
            return true;
        }
    };
}

#endif 

