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
        {1, 2, 3, 4, 5,
         2, 3, 4, 5, 1, 
         3, 4, 5, 1, 2, 
         4, 5, 1, 2, 3,
         5, 1, 2, 3, 4};
    shape = {5, 5};
    auto mat1 = ndarray<double>(data,shape);
    mat1.show();

    startTime = clock();
    auto eigvals = nc.linaig.eigvals(mat1);
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    eigvals.show();

    return 0;
}