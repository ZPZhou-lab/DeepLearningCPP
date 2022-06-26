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
    vector<int> strides;;

    shape = {3,2,4};;
    auto mat1 = nc.arange<double>(0,24);
    mat1 = mat1.reshape(shape);
    
    axis = {1};
    startTime = clock();
    auto mat2 = mat1.all(axis);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    mat2.show();

    return 0;
}