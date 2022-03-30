#include<bits/stdc++.h>
#include "ndarray.cpp"
#include<vector>
#include <typeinfo>
#include<random>
#include<chrono>
#pragma once
using namespace std;

static set<string> int_dtypes = {"i","l","x"};
static set<string> real_dtypes = {"f","d","e"};

class numcpp{
public:
    // 创建零矩阵
    template <typename T>
    ndarray<T> zeros(vector<long>& shape);

    // 创建全1矩阵
    template <typename T>
    ndarray<T> ones(vector<long>& shape);

    // 创建均匀分布随机矩阵
    template <typename T>
    ndarray<T> rand(double low, double high, vector<long>& shape);

    // 创建正态分布随机矩阵
    template <typename T>
    ndarray<T> normal(double mean, double scale, vector<long>& shape);
    template <typename T> // 换个名字
    ndarray<T> randn(double mean, double scale, vector<long>& shape);
};


// 构造零矩阵
template <typename T>
ndarray<T> numcpp::zeros(vector<long>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    vector<T> arr(size,0);
    ndarray<T> mat(arr,shape);
    return mat;
}

// 构造全1矩阵
template <typename T>
ndarray<T> numcpp::ones(vector<long>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    vector<T> arr(size,1);
    ndarray<T> mat(arr,shape);
    return mat;
}


// 构造均匀分布矩阵
template <typename T>
ndarray<T> numcpp::rand(double low, double high, vector<long>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    typedef T value_type; // 定义类别属性
    value_type _dtype; // 数据类型
    const type_info &dataInfo = typeid(_dtype);
    // 创建随机数生成器
    unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // 随机数种子
    default_random_engine generator(seed);
    uniform_int_distribution<long> distribution_int(low,high); // 整型发生器
    uniform_real_distribution<double> distribution_real(low,high); // 浮点型发生器
    vector<T> arr(size);
    if(int_dtypes.find(dataInfo.name()) != int_dtypes.end()){
        auto dice = bind(distribution_int,generator);
        for(long long i=0;i<size;++i) arr[i] = dice();
    }else{
        auto dice = bind(distribution_real,generator);
        for(long long i=0;i<size;++i) arr[i] = dice();
    }
    ndarray<T> mat(arr,shape);
    return mat;
}

// 构造正态分布矩阵
template <typename T>
ndarray<T> numcpp::normal(double mean, double scale, vector<long>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    // 创建随机数生成器
    unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // 随机数种子
    default_random_engine generator(seed);
    normal_distribution<double> distribution(mean,scale); // 浮点型发生器
    vector<T> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<T> mat(arr,shape);
    return mat;
}
// 构造正态分布矩阵的重构
template <typename T>
ndarray<T> numcpp::randn(double mean, double scale, vector<long>& shape){
    return this->normal<T>(mean,scale,shape);
}