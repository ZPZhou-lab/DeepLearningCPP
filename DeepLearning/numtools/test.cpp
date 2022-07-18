#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <vector>
using namespace std;


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> axis;
    vector<int> strides;


    shape = {3,5};
    auto mat1 = nc.random.standard_cauchy(shape);
    mat1.show();
    auto mat2 = nc.random.randn(5,4);
    mat2.show();
    

    startTime = clock();
    auto mat3 = mat1.dot(mat2);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    mat3.show();

    return 0;
}