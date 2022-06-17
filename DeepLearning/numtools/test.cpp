#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <vector>
#include <signal.h>
using namespace std;


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> axis;
    vector<int> strides;

    shape = {3,2,4};
    ndarray<double> mat = nc.arange<double>(1, 25);
    mat = mat.reshape(shape);

    mat.show();
\
    mat = mat / 0.0;
    mat.show();

    return 0;
}