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
    
    vector<double> data = {1,0.1,0.1,1};
    shape = {2,2};
    auto mat1 = ndarray<double>(data,shape);
    // mat1.show();
    // b.show();

    startTime = clock();
    auto L = nc.linaig.cholesky(mat1);
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    L.show();

    auto L_ = L.T();
    L_.show();

    auto prod = L.dot(L_);
    prod.show();

    return 0;
}