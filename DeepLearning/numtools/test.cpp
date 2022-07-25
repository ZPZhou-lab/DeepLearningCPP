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

    auto mat1 = nc.random.randn(2,300,4,200);
    auto mat2 = nc.sum(mat1) / mat1.size();
    cout<<"sum of mat1: "<<mat2<<endl;
    
    startTime = clock();
    axis = {1,3};
    auto mat3 = nc.sum(mat1,axis,true);
    mat3 = mat3 / mat3.size();
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    mat3.show();

    return 0;
}