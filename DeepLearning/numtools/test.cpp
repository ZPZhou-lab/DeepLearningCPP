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
    
    shape = {500,800};
    ndarray<double> mat1 = nc.randn<double>(shape);
    axes = {1,0};
    mat1 = mat1.transpose(axes);

    shape = {80,500,10};

    startTime = clock();
    mat1 = mat1.reshape(shape);
    endTime = clock();
    printf("time used: %.4fS\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    // ndarray<double> mat2 = nc.arange<double>(-25, -1);
    // shape = {4,3,2};
    // mat2 = mat2.reshape(shape);

    // axes = {0,2,1};
    // mat1 = mat1.transpose(axes);
    // axes = {1,2,0};
    // mat2 = mat2.transpose(axes);

    // mat1.show();
    // mat2.show();

    // auto mat3 = mat1 / 2;
    // mat3.show();

    return 0;
}