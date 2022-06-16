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

    shape = {10,5,4,25};
    ndarray<double> mat = nc.arange<double>(1, 5001);
    mat = mat.reshape(shape);

    axis = {0,3};

    double sum1 = mat.sum();
    ndarray<double> sum2 = mat.sum(axis);
    double mean1 = mat.mean();
    // ndarray<double> mean2 = mat.mean(axis);
    cout<<"sum: "<<sum1<<endl;
    // cout<<"sum: "<<sum2.item(0)<<endl;
    cout<<"mean: "<<mean1<<endl;

    sum2.show();

    return 0;
}