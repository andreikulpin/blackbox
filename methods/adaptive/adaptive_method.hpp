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
     * Adaptive method implementation
     * 1. Заранее вычисляется M направлений, которые выходят из исходной точки и соединяют ее с точками,
     * равномерно распределенными на сфере ( в функции direction мы получаем вектора)
     * 2. Ищем среди них наилучшее направление (максимально уменьшающее значение функции) таким образом:
     *      проверяем каждое следующее направление на уменьшение значения целевой функции (в сравнении с текущей),
     *      если значение в найденной точке уменьшается, то делаем в этом направлении увеличенный шаг и смотрим, как это влияет на 
     *      итоговое значение. Если оно стало меньше, тогда сравниваем его с остальными и запоминаем лучшее. 
     * Если такое направление находится, то увеличиваем шаг и генерируем новые направления, в противном случае шаг уменьшаем
     * и генерируем новые направления
     * 3. Поиск продолжаем, пока размер шага не станет совсем незначительным
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
            int numbOfPoints = 10;
            /**
             * Minimal value of step
             */
            FT minStep = 1e-08;
            /**
             * Increase in the case of success
             */
            FT mInc = 1.618;
            /**
             * Decrease in the case of failure
             */
            FT mDec = 0.618;
            /**
             * Upper and lower bound 
             */
            FT mHUB = 1e+02;
            FT mHLB = 1e-08;
            /**
             * Trace on/off
             */
            bool mDoTracing = true;
            /**
             * Max steps number
             */
            int maxStepNumber = 100;
        };

        FT search(int n, FT* x, const FT* leftBound, const FT* rightBound, const std::function<FT(const FT*)> &f) override {

            double v;
            FT fcur = f(x);
            int StepNumber = 0; 
            bool br = false;
            FT* dirs;
            FT sft = 1.0;

            dirs = new FT[n * mOptions.numbOfPoints];
            
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<FT> distribution(0.0,1.0);

            auto direction = [&] (int amount_of_points) {
                for (int j = 0; j < amount_of_points; j++) {
                    FT sum = 0.0;
                    //get vector of gaussian variables and find the sum for normalization
                    for (int i = 0; i< n; i++)
                    {
                        dirs [n*j + i] = distribution(generator);
                        sum += (dirs [n*j + i]) * (dirs [n*j + i]);  
                    }
                    sum = sqrt(sum);
                    //normalize direction for uniform on M-sphere
                    for (int i = 0; i< n; i++)
                    {
                        dirs [n*j + i] /= sum;  
                    }
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
                //t = SGMAX(t, mOptions.mHLB);
                return t;
            };

            auto step = [&] () {
                bool isStepSuccessful = false;
                const FT h = sft;
                const double e = 2.718281828;

                    bool dirty_flag = false;
                    FT f_best[mOptions.numbOfPoints];
                    snowgoose::VecUtils::vecSet(mOptions.numbOfPoints, fcur + 1.0, f_best);
                    //FT best_f = fcur;
                    FT x_best[n];
                    int numb_of_best_vec = 0.0;
                #pragma omp parallel for
                for (int i = 0; i < mOptions.numbOfPoints; i++) 
                    {
                        FT xtmp[n]; 
                        //calculation of new point
                        for (int j = 0; j < n; j++)
                        {
                            xtmp[j] = x[j] + dirs[i * n + j] * h;
                        }
                        //if (!isInBox(n, xtmp, leftBound, rightBound)) continue;
                        int box_flag = isInBox(n, xtmp, leftBound, rightBound);
                        if ( box_flag != 0) 
                        {
                            box_flag == 1 ? snowgoose::VecUtils::vecCopy(n, rightBound, xtmp) : snowgoose::VecUtils::vecCopy(n, leftBound, xtmp);;
                        }

                        FT fn = f(xtmp);
                        //if value in this point less than previous one, trying to make a bigger step in this direction
                        if (fn < fcur) {
                            dirty_flag = true;
                            FT x_continued[n];
                            snowgoose::VecUtils::vecSaxpy(n, xtmp, x, -1.0, x_continued);
                            snowgoose::VecUtils::vecSaxpy(n, x, x_continued, mOptions.mInc, x_continued);
                            //if (!isInBox(n, x_continued, leftBound, rightBound)) continue;
                            int box_flag1 = isInBox(n, x_continued, leftBound, rightBound);
                            if ( box_flag1 != 0) 
                            {
                                box_flag1 == 1 ? snowgoose::VecUtils::vecCopy(n, rightBound, x_continued) : snowgoose::VecUtils::vecCopy(n, leftBound, x_continued);;
                            }
                            
                            FT f_continued = f(x_continued);
                            if (f_continued < fn) f_best[i] = f_continued;
                            else f_best[i] = fn;
                        }
                    }

                //save new x vector    
                if (dirty_flag) 
                    {
                        fcur = snowgoose::VecUtils::min(mOptions.numbOfPoints, f_best, &numb_of_best_vec);
                        //calculation of x
                        FT xtmp[n]; 
                        for (int j = 0; j < n; j++)
                        {
                            xtmp[j] = x[j] + dirs[ numb_of_best_vec * n + j] * h;
                        }
                        FT fn = f(xtmp);
                        if (fn != fcur) {
                            FT x_continued[n];
                            snowgoose::VecUtils::vecSaxpy(n, xtmp, x, -1.0, x_continued);
                            snowgoose::VecUtils::vecSaxpy(n, x, x_continued, mOptions.mInc, x_continued);
                            snowgoose::VecUtils::vecCopy(n, x_continued, x);
                        }
                        else snowgoose::VecUtils::vecCopy(n, xtmp, x);

                        isStepSuccessful = true;
                        return isStepSuccessful;
                    }
                else return isStepSuccessful;
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
                
                if (StepNumber >= mOptions.maxStepNumber) br = true;
                else {
                    if (!success) {
                        if (sft > mOptions.minStep) sft = dec(sft);
                        else 
                            br = true;
                    } else sft = inc(sft);
                }   
                
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

        int isInBox(int n, const FT* x, const FT* a, const FT* b) const {
            for (int i = 0; i < n; i++) {
                if (x[i] > b[i]) {
                    return 1;
                }

                if (x[i] < a[i]) {
                    return -1;
                }
            }
            return 0;
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

