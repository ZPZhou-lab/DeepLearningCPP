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

    shape = {300,200,400};
    ndarray<double> mat = nc.randn<double>(shape);

    axis = {0,2};



    double sum1 = mat.sum();
    ndarray<double> sum2 = mat.sum(axis);
    double mean1 = mat.mean();

    startTime = clock();
    ndarray<double> mean2 = mat.mean(axis);
    endTime = clock();

    printf("time used: %.4f\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    cout<<"sum: "<<sum1<<endl;
    // cout<<"sum: "<<sum2.item(0)<<endl;
    cout<<"mean: "<<mean1<<endl;

    return 0;
}