/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   bruteforce.hpp
 * Author: mikhail
 *
 * Created on February 22, 2019, 10:04 PM
 */

#ifndef BRUTEFORCE_HPP
#define BRUTEFORCE_HPP

#include <math.h>
#include <algorithm>
#include <limits>
#include <common/bbsolver.hpp>

/**
 * A simple rectangular mesh black-box optimizer
 */
template <class T> class BruteForce : public BlackBoxSolver <T> {
public:

    /**
     * Constructor
     * @param p number of mesh points per dimension
     */
    BruteForce(int p) : mP(p) {
    }

    T search(int n, T* x, const T * const a, const T * const b, const std::function<T(const T * const)> &f) {
        const int tot = pow(mP, n);
        T *y = new T[n];
        T fr = std::numeric_limits<T>::max();
        for(int i = 0; i < tot; i ++) {
            for(int j = 0; j < n; j ++) {
                int I = i;
                y[j] = a[j] + (T)((I - (I / mP) * mP)) * (b[j] - a[j]) / (T) mP;
                I = I / mP;
            }
            T v = f(y);
            if(v < fr) {
                std::cout << "upd: " << v << "\n";
                fr = v;
                std::copy(y, y + n, x);
            }
        }
        delete [] y;
        return fr;
    }

private:
    int mP;

};

#endif /* BRUTEFORCE_HPP */

