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

    auto mat1 = nc.random.randn(5);
    vector<double> p = {10,1,10,1,1};
    mat1.show();
    shape = {5,7};
    auto mat2 = nc.random.choice(mat1,shape,true,p);
    mat2.show();
    

    startTime = clock();
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    return 0;
}