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

    auto mat1 = nc.random.randn(8,8);
    mat1.show();

    startTime = clock();
    auto lu = nc.linaig.LU(mat1);
    auto mat2 = lu.first;
    auto mat3 = lu.second;
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    
    cout<<"L: "<<endl;
    mat2.show();
    cout<<"U: "<<endl;
    mat3.show();

    auto diff = mat2.dot(mat3) - mat1;
    diff.show();

    return 0;
}