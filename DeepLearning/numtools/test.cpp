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
    vector<int> strides;;
    
    vector<double> arr = {1,2,3,4};
    shape = {2,2};
    ndarray<double> mat(arr,shape);

    mat.show();

    arr[2] = 10;
    mat.show();

    return 0;
}