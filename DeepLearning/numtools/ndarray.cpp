#include<bits/stdc++.h>
#include<vector>
#include <typeinfo>
#include<stdarg.h>
#pragma once
using namespace std;

template <typename T>
class ndarray{
private:
    typedef T value_type; // 定义类别属性
    vector<long> _shape; // 矩阵维度
    vector<T> data; // 数据
    value_type _dtype; // 数据类型
    int _ndim; // 维度数量
    long long _size; // 元素个数
    vector<long long> _idx_prod; // 存在数组维度的累计乘积

public:
    // 默认构造函数
    ndarray();
    // 构造函数
    ndarray(vector<T>& arr, vector<long>& shape);
    // 析构函数
    ~ndarray();
    
    // 访问元素
    template <typename ...Args>
    T at(Args...args);

    // 用序号访问元素
    T iloc(long long idx);

    // 返回数据类型
    string dtype(void);
    // 返回元素个数
    long long size(void);
    // 返回矩阵维度
    int ndim(void);
    // 矩阵形状
    vector<long> shape(void);

};

// 从不定长参数列表中获取参数
template <typename T>
vector<T> fetchArgs(vector<T>& fetch, T arg){
    fetch.emplace_back(arg);
    return fetch;
}
template <typename T, typename ...Args>
vector<T> fetchArgs(vector<T>& fetch, T arg, Args...args){
    fetch.emplace_back(arg);
    return fetchArgs(fetch,args...);
}

// 默认构造函数
template <typename T>
ndarray<T>::ndarray(){
    this->_ndim = 0;
    this->_size = 0;
}

// 构造函数
template <typename T>
ndarray<T>::ndarray(vector<T>& arr, vector<long>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    assert((long long)arr.size() == size);
    this->data = arr;
    this->_shape = shape;
    this->_size = size;
    this->_ndim = shape.size();
    // 统计累计索引
    vector<long long> idx_prod(this->_ndim,1);
    for(int i=this->_ndim-1;i>0;--i) idx_prod[this->_ndim-i] = idx_prod[this->_ndim-i-1] * this->_shape[i];
    this->_idx_prod = idx_prod;
}

//  析构函数
template <typename T>
ndarray<T>::~ndarray(){
    cout<<"Destructor called"<<endl;
}

// 获取元素的值
template <typename T>
template <typename ...Args>
T ndarray<T>::at(Args...args){
    // 取出索引
    vector<int> idx;
    idx = fetchArgs(idx,args...);
    // 判断维度是否正确
    assert((int)idx.size() == this->_ndim);
    long long loc = 0;
    for(int i=0;i<this->_ndim;++i) loc += this->_idx_prod[this->_ndim-i-1]*idx[i];
    assert(loc >= 0 && loc < this->_size);
    return this->data[loc];
}

// 用序号访问元素
template <typename T>
T ndarray<T>::iloc(long long idx){
    // 取出索引
    return this->data[idx];
}

// 数据类型
template <typename T>
string ndarray<T>::dtype(void){
    map<string,string> dtypes = {
        {"i","int"}, {"f","float"}, {"d","double"}, {"l","long"}, {"b","bool"},
        {"e", "long double"}, {"x","long long"}
    };
    const type_info &dataInfo = typeid(this->_dtype);
    return dtypes[dataInfo.name()];
}

// 元素个数
template <typename T>
long long ndarray<T>::size(void){
    return this->_size;
}

// 矩阵维度
template <typename T>
int ndarray<T>::ndim(void){
    return this->_ndim;
}

// 矩阵维度
template <typename T>
vector<long> ndarray<T>::shape(void){
    return this->_shape;
}