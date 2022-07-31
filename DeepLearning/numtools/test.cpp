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

    vector<double> data = 
        {1, 2, 3, 4, 
         2, 3, 4, 1, 
         3, 4, 1, 2, 
         4, 1, 2, 3};
    shape = {4,4};
    auto mat1 = ndarray<double>(data,shape);
    mat1.show();

    startTime = clock();
    auto eigs = nc.linaig.eig(mat1);
    auto eigVals = eigs.first;
    auto eigVecs= eigs.second;
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    eigVals.show();
    eigVecs.show();

    auto prod = mat1.dot(eigVecs);
    auto multi = eigVecs * eigVals;
    prod.show();
    multi.show();


    return 0;
}