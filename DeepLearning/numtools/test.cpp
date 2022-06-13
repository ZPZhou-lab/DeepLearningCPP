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

    shape = {80,50};
    ndarray<float> mat = nc.normal<float>(0, 1, shape);
    axes = {1,0};
    mat = mat.transpose(axes);

    shape = {1,5,10,1,8,1,10};
    mat = mat.reshape(shape);

    axis = {0,3,5};
    mat = mat.squeeze(axis);
    shape = mat.shape();
    for(auto s:shape) cout<<s<<" ";
    cout<<endl;

    return 0;
}