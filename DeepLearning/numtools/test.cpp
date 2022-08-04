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

    auto mat1 = nc.random.rand(100,100);
    auto b = nc.random.rand(100,1);
    // mat1.show();
    // b.show();

    startTime = clock();
    auto x = nc.linaig.solve(mat1,b);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    auto diff = mat1.dot(x) - b;
    cout<<diff.max()<<endl;

    return 0;
}