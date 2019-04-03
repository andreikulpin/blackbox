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

#ifndef GRANULAR_METHOD_HPP
#define GRANULAR_METHOD_HPP

#include <sstream>
#include <stdlib.h>
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
#include <math.h>
#include <string>
#include <chrono>
#include <random>

namespace LOCSEARCH {
        /**
     * Random method implementation
     * 1. Заранее задается t - число, лежащее в интервале от -0.5 до 0.5 и grain_size, равное 1.0
     * 2. Вычисляется n направлений, каждое из которых - единичное, совпадающее с одной из n осей по направлению
     * 3. Каждый шаг рандомно выбирается одно из этих направлений и в эту сторону делается шаг размером t * grain_size
     * 4. Если значение функции уменьшается в данном направлении - запоминаем его и снова выбираем направление
     * 5. Каждые 100 итераций проверяем число неудачных шагов - в случае, если их было не меньше 95%,
     * уменьшаем grain_size 
     * 6. Поиск продолжаем, пока не закончится заданное заранее число шагов
     */
    template <typename FT> class GranularMethod : public BlackBoxSolver<FT> {

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
             * Trace on/off
             */
            bool mDoTracing = false;
            /**
             * Max steps number
             */
            int maxStepNumber = 1000;
        };

        /**
         * The constructor
         * @param prob - reference to the problem
         * @param stopper - reference to the stopper
         * @param ls - pointer to the line search
         */
        /**
         * Perform search
         * @param x start point and result
         * @param v  the resulting value
         * @return true if search converged and false otherwise
         */
      
        FT search(int n, FT* x, const FT* leftBound, const FT* rightBound, const std::function<FT ( const FT* )> &f) override {
            
            double v;
            FT fcur = f(x);
            int StepNumber = 0; 
            int Unsuccess = 0;
            double grain_size = 1.0;
            bool br = false;
            FT* dirs;
            FT sft = 1.0;
            dirs = new FT[n*n];
            
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            std::normal_distribution<FT> distribution(0.0,1.0);
            std::mt19937_64 rng;
            uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
            rng.seed(ss);
            std::uniform_real_distribution<double> unif(0, 1);
            

            auto direction = [&] (int amount_of_points) {
                snowgoose::VecUtils::vecSet(n * n, 0., dirs);
                for (int i = 0; i < n; i++) 
                {
                    dirs[i * n + i] = 1;
                }
            };

            auto step = [&] () {
                bool isStepSuccessful = false;
                int Unsuccess = 0;
                const double e = 2.718281828;

                    FT* parameter_tweak = new FT[n];
                    double t = unif(rng) - 0.5;
                    FT xtmp[n];
                    int vector_number = rand() % n; 
                        for (int j = 0; j < n; j++)
                        {
                            parameter_tweak[j] = t * dirs[vector_number * n + j] * grain_size;
                            xtmp[j] = parameter_tweak[j] + x[j];
                        }
                        if (isInBox(n, xtmp, leftBound, rightBound)) {
                            FT fn = f(xtmp);
                            if (fn < fcur)
                            {
                                isStepSuccessful = true;
                                snowgoose::VecUtils::vecCopy(n, xtmp, x);
                                fcur = fn;
                            }
                        }
                
                return isStepSuccessful;
            };
           
            direction(1);
            while (!br) {
                
                bool success = step();
 
                StepNumber++;
                if (mOptions.mDoTracing) {
                    std::cout << (success ? "Success" : "Not success") << std::endl;
                    std::cout << "f =" << fcur << std::endl;
                    std::cout << "sft =" << sft << std::endl;
                }
                
                if (!success) {
                    
                        ++Unsuccess; 
                        if ((StepNumber%100 == 0) && (Unsuccess >= 95))
                        {
                            grain_size *= 0.1;
                            Unsuccess = 0;
                        }
                           
                }
                
                if (StepNumber >= mOptions.maxStepNumber)   br = true;

                
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

        /**
         * Get watchers' vector
         * @return watchers vector
         
        std::vector<Watcher>& getWatchers() {
            return mWatchers;
        }*/

    private:

        //const COMPI::MPProblem<FT>& mProblem;
        Options mOptions;
        std::vector<Stopper> mStoppers;
        //std::vector<Watcher> mWatchers;
        //std::unique_ptr<LineSearch<FT>> mLS;
        FT mGlobMin;
        
        void printArray(int n, FT * array) const {
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
        
        void printVector(int n, std::vector<FT> vector) {
            std::cout << " dirs = ";
            for (int i = 0; i < n; i++) {
                std::cout << vector[i] << ", ";
            }
            std::cout << " ]" << std::endl;
        }
    };
};

#endif 

