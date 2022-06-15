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

    shape = {400,250};
    ndarray<double> mat = nc.randn<double>(shape);
    axes = {1,0};
    mat = mat.transpose(axes);

    double sum1 = mat.sum();
    ndarray<double> sum2 = mat.sum(axis);
    double mean1 = mat.mean();
    // ndarray<double> mean2 = mat.mean(axis);
    cout<<"sum: "<<sum1<<endl;
    cout<<"sum: "<<sum2.item(0)<<endl;
    cout<<"mean: "<<mean1<<endl;

    return 0;
}