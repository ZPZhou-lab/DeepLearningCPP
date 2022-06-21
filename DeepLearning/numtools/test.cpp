#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <vector>
using namespace std;


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> axis;
    vector<int> strides;;

    shape = {3,4};
    auto mat1 = nc.randn<double>(shape);
    mat1.show();

    mat1[0] = 100;
    mat1.show();
    
    return 0;
}