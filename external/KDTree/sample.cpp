#include <iostream>
#include <array>

#include "kdtree.h"

// user-defined point type
// inherits std::array in order to use operator[]
class MyPoint : public std::array<double, 3>
{
public:

	// dimension of space (or "k" of k-d tree)
	// KDTree class accesses this member
	static const int DIM = 3;

	// the constructors
	MyPoint() {}
	MyPoint(double x, double y, double z)
	{ 
		(*this)[0] = x;
		(*this)[1] = y;
        (*this)[2] = z;
	}

};
