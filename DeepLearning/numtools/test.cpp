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
    vector<int> strides;

    
    ndarray<double> mat1 = nc.arange<double>(1, 25);
    shape = {3,4,2};
    mat1 = mat1.reshape(shape);

    ndarray<double> mat2 = nc.arange<double>(-25, -1);
    shape = {4,3,2};
    mat2 = mat2.reshape(shape);

    axes = {0,2,1};
    mat1 = mat1.transpose(axes);
    axes = {1,2,0};
    mat2 = mat2.transpose(axes);

    auto mat3 = mat1 + mat2;
    mat3.show();

    return 0;
}