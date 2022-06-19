#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <vector>
#include <signal.h>
using namespace std;


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> axis;
    vector<int> strides;;
    
    ndarray<double> mat1 = nc.arange<double>(1, 25);
    shape = {3,2,4};
    mat1 = mat1.reshape(shape);

    ndarray<double> mat2 = nc.arange<double>(-25, -1);
    mat2 = mat2.reshape(shape);

    auto mat3 = mat1 / mat2;
    mat3.show();

    return 0;
}