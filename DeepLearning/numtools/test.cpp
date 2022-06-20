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

    shape = {800,500};
    ndarray<double> mat1 = nc.randn<double>(shape);    
    ndarray<double> mat2 = nc.randn<double>(shape);

    startTime = clock();
    auto mat3 = mat1 + mat2;
    endTime = clock();
    printf("Time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC );

    return 0;
}