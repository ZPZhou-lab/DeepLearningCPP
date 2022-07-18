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
    auto mat1 = nc.random.randint<long>(0,5,shape);
    mat1.show();
    shape = {5,4};
    auto mat2 = nc.random.randint<long long>(4,5,shape);
    mat2.show();
    

    startTime = clock();
    auto mat3 = mat1.dot(mat2);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    mat3.show();

    return 0;
}