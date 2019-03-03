#ifndef GRIDSOLVEROMP_HPP
#define GRIDSOLVEROMP_HPP

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <limits>
#include "gridsolver.hpp"


template <class T> class GridSolverOMP : public GridSolver <T> {
public:

	/**
	* Constructor
	*/
	GridSolverOMP() {
		np = omp_get_num_procs();
		omp_set_dynamic(0);
		omp_set_num_threads(np);
	}



protected:
	int np;
	void GridEvaluator(int dim, const T *a, const T *b, T* xfound, T *Frp, T *LBp, T *dL, const std::function<T(const T * const)> &compute) {
		T Fr = std::numeric_limits<T>::max(), L = std::numeric_limits<T>::min(), delta = std::numeric_limits<T>::min(), LB;
		double R;
		T *step, *x, *Fvalues, *Frs, *Ls;
		int *pts;
		step = new T[dim]; //step of grid in every dimension
		x = new T[dim]; //algorithm now needs to evaluate function in two adjacent point at the same time
		Frs = new T[np];
		Ls = new T[np];
		pts = new int[np];
		if ((step == nullptr) || (Frs == nullptr) || (Ls == nullptr) || (pts == nullptr)) {
			this->errcode = -2;
			return;
		}
		for (int k = 0; k < np; k++) {
			Frs[k] = std::numeric_limits<T>::max();
			Ls[k] = std::numeric_limits<T>::min();
		}
		for (int i = 0; i < dim; i++) {
			step[i] = fabs(b[i] - a[i]) / (this->nodes - 1);
			delta = step[i] > delta ? step[i] : delta;
		}
		delta = delta * 0.5 * dim;
		R = getR(delta);	//value of r
		int allnodes = static_cast<int>(pow(this->nodes, dim));
		int node;
		Fvalues = new T[allnodes];
		if (Fvalues == nullptr) {
			this->errcode = -2;
			return;
		}
#pragma omp parallel private(x) shared(dim, step, a, Fvalues)
		{
			x = new T[dim];
#pragma omp for
			for (int j = 0; j < allnodes; j++) {
				int point = j;
				for (int k = dim - 1; k >= 0; k--) {
					int t = point % this->nodes;
					point = (int)(point / this->nodes);
					x[k] = a[k] + t * step[k];
				}
				T rs = compute(x);
				Fvalues[j] = rs;
				int nt = omp_get_thread_num();
				if (rs < Frs[nt]) {
					Frs[nt] = rs;
					pts[nt] = j;
				}
			}
			delete[]x;
		}
		this->fevals += allnodes;


#pragma omp parallel for shared(dim,allnodes,Fvalues)
		for (int j = 0; j < allnodes; j++) {
			int neighbour;
			T loc = std::numeric_limits<T>::min();
			for (int k = 0; k < dim; k++) {
				int board = (int)pow(this->nodes, k + 1);
				neighbour = j + board / this->nodes;
				if ((neighbour < allnodes) && ((j / board) == (neighbour / board))) {
					loc = static_cast<T>(fabs(Fvalues[j] - Fvalues[neighbour])) / step[dim - 1 - k];
				}
				int nt = omp_get_thread_num();
				if (loc > Ls[nt]) {
					Ls[nt] = loc;
				}
			}
		}
		for (int k = 0; k < np; k++) {
			if (Frs[k] < Fr) {
				Fr = Frs[k];
				node = pts[k];
			}
			L = Ls[k] > L ? Ls[k] : L;
		}
		for (int k = dim - 1; k >= 0; k--) {
			int t = node % this->nodes;
			node = (int)(node / this->nodes);
			xfound[k] = a[k] + t * step[k];
		}
		//final calculation
		LB = R * L * delta;
		*dL = LB;
		LB = Fr - LB;
		delete[]step; delete[]x;
		delete[]Fvalues;
		*Frp = Fr;
		*LBp = LB;
	}

};


#endif /* GRIDSOLEROMP_HPP */
