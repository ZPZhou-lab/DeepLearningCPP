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

    auto mat1 = nc.random.randn(2,3,4,2);
    mat1.show();
    auto mat2 = nc.sum(mat1) / mat1.size();
    cout<<"sum of mat1: "<<mat2<<endl;
    
    startTime = clock();
    axis = {1,3};
    auto mat3 = nc.min(mat1,axis,true);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);
    cout<<"mat3: "<<endl;
    mat3.show();

    cout<<"mat4: "<<endl;
    auto mat4 = mat3.astype<float>();
    mat4.show();
    auto mat5 = nc.exp(mat4);
    mat5.show();
    auto mat6 = nc.log(mat5);
    mat6.show();

    return 0;
}