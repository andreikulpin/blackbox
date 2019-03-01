/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#include <iterator>
#include "bruteforce.hpp"

constexpr int n = 3;

double f(const double* x) {
    double v;
    for(int i = 0; i < n; i ++)
        v += x[i] * x[i];
    return v;
}
int main() {
    BruteForce<double> bf(16);
    double x[n];
    double a[n], b[n];
    std::fill(a, a + n, -1);
    std::fill(b, b + n, 2);
    double v = bf.search(n, x, a, b, f);
    std::cout << "Found " << v << " at [" ;
    std::copy(x, x + n, std::ostream_iterator<double>(std::cout, " "));
    std::cout << "]\n";
}