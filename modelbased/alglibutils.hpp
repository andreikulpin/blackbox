
#include "ap.h"

namespace AL {
    void printArray(alglib::real_1d_array a) {
	for (int i = 0; i < a.length(); i++) {
		std::cout << a(i) << " ";
	}
	std::cout << std::endl;
}

    void printMatrix(alglib::real_2d_array matrix) {
	for (int i = 0; i < matrix.rows(); i++) {
		for (int j = 0; j < matrix.cols(); j++) {
			std::cout << matrix(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
    }
}

