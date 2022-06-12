#include "ndarray.cpp"
#include "numcpp.cpp"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime>
#include <vector>
#include <signal.h>
using namespace std;


int main(){
    cout<<"Compiled Successfully"<<endl;
    clock_t startTime, endTime;
    numcpp nc;

    // 测试1
    cout<<"\n"<<"Test1"<<endl;
    vector<double> arr = {
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0
    };
    vector<int> shape1 = {2,3,2};
    ndarray<double> mat1(arr, shape1);
    cout<<"dtype: "<<mat1.dtype()<<endl;
    cout<<"ndim: "<<mat1.ndim()<<endl;
    cout<<"size: "<<mat1.size()<<endl;
    cout<<mat1.at(0,2,0)<<endl;
    cout<<mat1.item(1)<<endl;
    mat1.show();
    
    // 测试2
    cout<<"\n"<<"Test2"<<endl;
    vector<int> shape2 = {2,3,4};
    ndarray<double> mat2 = nc.rand<double>(0,5,shape2);
    cout<<"dtype: "<<mat2.dtype()<<endl;
    cout<<"ndim: "<<mat2.ndim()<<endl;
    cout<<"size: "<<mat2.size()<<endl;
    cout<<mat2.at(0,2,0)<<endl;
    cout<<mat2.item(9)<<endl;

    ndarray<double> mat3 = nc.randn<double>(shape2);
    cout<<"dtype: "<<mat3.dtype()<<endl;
    cout<<mat3.at(0,2,0)<<endl;
    cout<<mat3.item(9)<<endl;

    // 测试3
    cout<<"\n"<<"Transpose Test"<<endl;
    vector<int> shape3 = {8000,5000};
    ndarray<int> mat4 = nc.arange<int>(0, 40000000);
    mat4 = mat4.reshape(shape3);

    // mat4.show();

    printf("Transpose Using Fast Rotating Shaft: \n");
    printf("Number of elements: %lld\n", mat4.size());
    vector<int> axes = {1,0};
    startTime = clock();
    mat4 = mat4.transpose(axes);
    // mat4.show();
    endTime = clock();
    printf("Time used %.6fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);

    // 等距格点测试
    ndarray<double> mat5 = nc.arange<double>(0,10);
    mat5.show();
    ndarray<double> mat6 = nc.linspace<double>(0,1,10);
    mat6.show();

    mat5 = nc.arange<double>(0,20);
    vector<int> shape4 = {2,5,2};
    mat5 = mat5.reshape(shape4);
    vector<int> axes2 = {2,0,1};
    mat5 = mat5.transpose(axes2);
    mat5.show();
    shape4 = {5,4};
    mat5 = mat5.reshape(shape4);
    mat5.show();

    shape4 = {500,400,200};
    startTime = clock();
    mat4 = mat4.reshape(shape4);
    // mat4.show();
    endTime = clock();
    printf("Time used %.6fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    printf("item 999 %d\n",mat4.item(999));

    return 0;
}