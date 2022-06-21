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

    shape = {3,2,4};
    auto mat1 = nc.arange<double>(1, 25);
    mat1 = mat1.reshape(shape);
    mat1.show();

    axis = {0};
    auto mat2 = mat1.min(axis,true);
    mat2.show();

    cout<<mat1.max()<<endl;
    
    return 0;
}