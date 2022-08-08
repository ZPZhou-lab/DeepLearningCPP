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
    
    // vector<double> data = {1,0.1,0.1,1};
    // shape = {2,2};
    // auto mat1 = ndarray<double>(data,shape);
    auto mat1 = nc.random.randn(8,8);
    auto mat2 = mat1.T();
    mat1 = mat1.dot(mat2);
    mat1.show();


    startTime = clock();
    auto L = nc.linaig.cholesky(mat1);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    L.show();

    auto L_ = L.T();
    
    auto diff = L.dot(L_) - mat1;
    diff.show();

    return 0;
}
