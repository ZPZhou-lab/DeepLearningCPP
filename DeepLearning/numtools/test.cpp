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
    auto mat1 = nc.random.randn(4,4);

    startTime = clock();
    auto mat_norm = nc.linaig.norm(mat1);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    cout<<"norm: "<<mat_norm<<endl;

    return 0;
}
