/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   best_point_method.hpp
 * Author: kate
 *
 * Created on 27 декабря 2018 г., 22:15
 */

#ifndef ADAPTIVE_METHOD_HPP
#define ADAPTIVE_METHOD_HPP

#include <sstream>
#include <stdlib.h>
#include <vector>
#include <functional>
#include <memory>
#include <common/bbsolver.hpp>
#include <common/vec.hpp>
#include <math.h>
#include <string>
#include <chrono>
#include <random>

namespace LOCSEARCH {

    /**
     * Random method implementation
     */

    template <typename FT> class AdaptiveMethod : public BlackBoxSolver<FT> {
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
         * @param stpn step number
         * @param gran - current granularity vector
         */
        using Watcher = std::function<void(FT fval, const FT* x, const std::vector<FT>& gran, int stpn) >;

        /**
         * Options for Gradient Box Descent method
         */
        struct Options {
            /**
             * Amount of points (better 3n) and max of unsuccessful steps
             */
            int numbOfPoints = 600;
            /**
             * Minimal value of step
             */
            FT minStep = 0.0000001;
            /**
             * Initial value of granularity
             */
            FT* mHInit = nullptr;
            /**
             * Increase in the case of success
             */
            FT mInc = 1.618;
            /**
             * Decrease in the case of failure
             */
            FT mDec = 0.618;
            /**
             * Lower bound for granularity
             */
            FT mHLB = 1e-08;
            /**
             * Upper bound on granularity
             */
            FT mHUB = 1e+02;
            /**
             * Trace on/off
             */
            bool mDoTracing = true;
            /**
             * Max steps number
             */
            int maxStepNumber = 800;
        };

        FT search(int n, FT* x, const FT* leftBound, const FT* rightBound, const std::function<FT(const FT*)> &f) override {

            double v;
            FT fcur = f(x);
            int StepNumber = 0;
            bool br = false;

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<FT> distrib(0.0, 1.0);
            std::mt19937_64 rng;
            uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
            rng.seed(ss);
            std::uniform_real_distribution<double> unif(0, 1);


            FT* dirs;
            FT* main_dir;
            FT sft = 1.0;

            dirs = new FT[n * mOptions.numbOfPoints];

            auto direction = [&] (int amount_of_points) {
                for (int j = 0; j < amount_of_points; j++) {
                    FT sum = 0.0;
                    for (int i = 0; i < n; i++) {
                        dirs [n * j + i] = distrib(generator);
                        sum += (dirs [n * j + i]) * (dirs [n * j + i]);
                    }
                    sum = sqrt(sum);
                    for (int i = 0; i < n; i++) {
                        dirs [n * j + i] /= sum;
                    }
                }
            };

            auto normalize = [&] () {
                FT sum = 0.0;
                for (int i = 0; i < n; i++) {
                    sum += (main_dir [i]) * (main_dir [i]);
                }
                sum = sqrt(sum);
                for (int i = 0; i < n; i++) {
                    main_dir [ i] /= sum;
                }
            };

            auto inc = [this] (FT h) {
                FT t = h;
                t = h * mOptions.mInc;
                t = SGMIN(t, mOptions.mHUB);
                return t;
            };

            auto dec = [this](FT h) {
                FT t = h;
                t = h * mOptions.mDec;
                t = SGMAX(t, mOptions.mHLB);
                return t;
            };

            auto step = [&] () {
                bool isStepSuccessful = false;
                const FT h = sft;

                FT best_f = fcur;
                FT x_best[n];
                int numb_of_best_vec = -1;

                for (int i = 0; i < mOptions.numbOfPoints; i++) {
                    bool global_continue = false;
                    FT xtmp[n];
                    for (int j = 0; j < n; j++) {
                        xtmp[j] = x[j] + dirs[i * n + j] * h;
                        if (!isInBox(n, xtmp, leftBound, rightBound)) {
                            global_continue = true;
                            break;
                        }
                    }
                    if (global_continue) {
                        continue;
                    }

                    if (isInBox(n, xtmp, leftBound, rightBound)) {
                        FT fn = f(xtmp);

                        if (fn < fcur) {
                            FT x_continued[n];
                            for (int q = 0; q < n; q++) {
                                x_continued[q] = x[q] + mOptions.mInc * (xtmp[q] - x[q]);
                                if (!isInBox(n, x_continued, leftBound, rightBound)) {
                                    global_continue = true;
                                    break;
                                }
                            }
                            if (global_continue) {
                                continue;
                            }
                            FT f_continued = f(x_continued);
                            if (f_continued < fcur) {
                                if (f_continued < best_f) {
                                    best_f = f_continued;
                                    snowgoose::VecUtils::vecCopy(n, x_continued, x_best);
                                    numb_of_best_vec = i;
                                }
                            }
                        }
                    }
                }
                if (numb_of_best_vec != -1) {
                    isStepSuccessful = true;
                    snowgoose::VecUtils::vecCopy(n, x_best, x);
                    fcur = best_f;
                }

                return isStepSuccessful;
            };

            while (!br) {

                direction(mOptions.numbOfPoints);

                bool success = step();

                StepNumber++;
                if (mOptions.mDoTracing) {
                    std::cout << (success ? "Success" : "Not success") << std::endl;
                    std::cout << "f =" << fcur << std::endl;
                    std::cout << "sft =" << sft << std::endl;
                }

                if (!success) {

                    if (sft > mOptions.minStep) {
                        sft = dec(sft);
                    }
                    else {
                        br = true;
                    }

                }

                if (StepNumber >= mOptions.maxStepNumber) {
                    br = true;
                } else sft = inc(sft);

                for (auto s : mStoppers) {
                    if (s(fcur, x, StepNumber)) {
                        br = true;
                        break;
                    }
                }
            }
            v = fcur;
            delete [] dirs;
            return v;
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

    private:

        Options mOptions;
        std::vector<Stopper> mStoppers;

        void printArray(int n, FT * array) {
            std::cout << " dirs = ";
            std::cout << snowgoose::VecUtils::vecPrint(n, array) << std::endl;
        }

        bool isInBox(int n, const FT* x, const FT* a, const FT* b) const {
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

        void printVector(int n, std::vector<FT> vector) const {
            std::cout << " dirs = ";
            for (int i = 0; i < n; i++) {
                std::cout << vector[i] << ", ";
            }
            std::cout << " ]" << std::endl;
        }
    };
}

#endif 

