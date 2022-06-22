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

    shape = {5,8,10};
    auto mat1 = nc.randn<double>(shape);
    
    startTime = clock();
    auto mat2 = mat1.argmin(1);
    endTime = clock();
    printf("time used: %.4f\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    mat2.show();

    return 0;
}