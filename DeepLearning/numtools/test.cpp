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
    
    shape = {5000,8000};
    ndarray<double> mat1 = nc.randn<double>(shape);
    axes = {1,0};
    axis = {0,2};
    mat1 = mat1.transpose(axes);

    shape = {800,500,100};

    startTime = clock();
    mat1 = mat1.reshape(shape);
    auto matsum = mat1.sum(axis);
    endTime = clock();
    printf("time used: %.4fS\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    ndarray<double> mat2 = nc.arange<double>(-25, -1);
    shape = {3,4,2};
    mat2 = mat2.reshape(shape);

    ndarray<double> mat3 = nc.arange<double>(1, 25);
    shape = {4,3,2};
    mat3 = mat3.reshape(shape);

    axes = {0,2,1};
    mat2 = mat2.transpose(axes);
    axes = {1,2,0};
    mat3 = mat3.transpose(axes);


    auto mat4 = mat2 + mat3;
    mat4.show();

    return 0;
}