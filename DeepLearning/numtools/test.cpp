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

    auto mat1 = nc.random.randn(50,50);
    // mat1.show();
    // b.show();

    startTime = clock();
    double det = nc.linaig.det(mat1);
    cout<<"det of mat1: "<<det<<endl;
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);



    return 0;
}