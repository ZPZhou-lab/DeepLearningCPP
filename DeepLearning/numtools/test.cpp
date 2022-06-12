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

    ndarray<int> mat = nc.arange<int>(0, 40000000);
    shape = {50,1,10,1,10,1,80,1,100};
    mat = mat.reshape(shape);
    axes = {1,3,0,2,8,6,4,5,7};
    mat = mat.transpose(axes);

    shape = mat.shape();
    for(auto s: shape) cout<<s<<"  ";
    cout<<endl;

    strides = mat.strides();
    for(auto s: strides) cout<<s<<"  ";
    cout<<endl;

    mat = mat.squeeze();

    shape = mat.shape();
    for(auto s: shape) cout<<s<<"  ";
    cout<<endl;

    strides = mat.strides();
    for(auto s: strides) cout<<s<<"  ";
    cout<<endl;
    // shape = {800,100,500};
    // mat = mat.reshape(shape);

    printf("item: %d\n",mat.item(10));

    return 0;
}