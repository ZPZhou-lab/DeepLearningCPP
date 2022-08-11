#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
using namespace std;


int main(){
    clock_t startTime, endTime;
    numcpp nc;

    vector<int> shape;
    vector<int> axes;
    vector<int> axis;
    vector<int> strides;
    
    // vector<double> data = {1,0.1,0.1,1};
    // shape = {2,2};
    // auto mat1 = ndarray<double>(data,shape);
    auto mat1 = nc.random.randn(40,40);

    startTime = clock();
    auto mat_2_norm = nc.linaig.norm(mat1,"2-norm");
    auto mat_1_norm = nc.linaig.norm(mat1,"1-norm");
    auto mat_f_norm = nc.linaig.norm(mat1,"f-norm");
    auto mat_inf_norm = nc.linaig.norm(mat1,"inf-norm");
    endTime = clock();
    printf("time used: %.4fs\n",(double)(endTime - startTime) / CLOCKS_PER_SEC);

    cout<<"2-norm: "<<mat_2_norm<<endl;
    cout<<"1-norm: "<<mat_1_norm<<endl;
    cout<<"f-norm: "<<mat_f_norm<<endl;
    cout<<"inf-norm: "<<mat_inf_norm<<endl;

    return 0;
}
