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

    shape = {3,2,4};
    auto mat1 = nc.randn<double>(shape);
    mat1.show();
    // mat2.show();

    startTime = clock();
    shape = {4,6};
    auto mat2 = nc.reshape(mat1, shape);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    mat2.show();

    return 0;
}