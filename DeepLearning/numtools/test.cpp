#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

    auto mat1 = nc.random.randn(2,3,1,2);
    auto mat2 = nc.random.randn(3,1);
    mat1.show();
    mat2.show();

    startTime = clock();
    ndarray<double> mat3 = mat1 - mat2;
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    mat3.show();
    return 0;
}