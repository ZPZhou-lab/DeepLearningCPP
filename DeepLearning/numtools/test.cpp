#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime>
#include <vector>
#include <signal.h>
using namespace std;


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> strides;

    ndarray<double> mat = nc.arange<double>(0, 40);
    shape = {5,8};
    mat = mat.reshape(shape);
    mat.show();

    cout<<mat.item(10)<<endl;

    return 0;
}