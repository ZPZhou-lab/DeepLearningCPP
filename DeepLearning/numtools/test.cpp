#include "ndarray.cpp"
#include<bits/stdc++.h>
using namespace std;

int main(){
    vector<double> arr = {
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0
    };
    vector<long> shape = {2,3,2};
    ndarray<double> mat(arr, shape);
    cout<<"Compiled Successfully"<<endl;
    cout<<"dtype: "<<mat.dtype()<<endl;
    cout<<"ndim: "<<mat.ndim()<<endl;
    cout<<"size: "<<mat.size()<<endl;
    cout<<mat.at(0,2,0)<<endl;
    cout<<mat.iloc(9)<<endl;
    return 0;
}