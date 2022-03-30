#include "ndarray.cpp"
#include "numcpp.cpp"
#include<bits/stdc++.h>
#include <chrono>
using namespace std;


int main(){
    cout<<"Compiled Successfully"<<endl;
    numcpp nc;

    // 测试1
    cout<<"\n"<<"Test1"<<endl;
    vector<double> arr = {
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0
    };
    vector<long> shape1 = {2,3,2};
    ndarray<double> mat1(arr, shape1);
    cout<<"dtype: "<<mat1.dtype()<<endl;
    cout<<"ndim: "<<mat1.ndim()<<endl;
    cout<<"size: "<<mat1.size()<<endl;
    cout<<mat1.at(0,2,0)<<endl;
    cout<<mat1.iloc(9)<<endl;
    
    // 测试2
    cout<<"\n"<<"Test2"<<endl;
    vector<long> shape2 = {2,3,4};
    ndarray<double> mat2 = nc.rand<double>(0,5,shape2);
    cout<<"dtype: "<<mat2.dtype()<<endl;
    cout<<"ndim: "<<mat2.ndim()<<endl;
    cout<<"size: "<<mat2.size()<<endl;
    cout<<mat2.at(0,2,0)<<endl;
    cout<<mat2.iloc(9)<<endl;

    ndarray<double> mat3 = nc.randn<double>(0,1,shape2);
    cout<<"dtype: "<<mat3.dtype()<<endl;
    cout<<mat3.at(0,2,0)<<endl;
    cout<<mat3.iloc(9)<<endl;

    // 测试3
    cout<<"\n"<<"Test3"<<endl;
    vector<long> shape3 = {2,5,4};
    ndarray<double> mat4 = nc.normal<double>(0,1,shape3);
    mat4.show();

    printf("Transpose: \n");
    vector<int> axes = {2,0,1};
    mat4 = mat4.transpose(axes);
    mat4.show();
    // 打印矩阵

    return 0;
}