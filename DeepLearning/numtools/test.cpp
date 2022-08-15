#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <complex>
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
    auto x = nc.random.randn(8,8);
    x.show();
    
    startTime = clock();
    auto Hx = nc.linaig.QR(x);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    auto Q = Hx.first, R = Hx.second;
    Q.show(), R.show();

    auto diff = Q.dot(R) - x;
    diff.show();

    auto Q_t = Q.T();
    auto Q_Q = Q.dot(Q_t);
    Q_Q.show();

    return 0;
}
