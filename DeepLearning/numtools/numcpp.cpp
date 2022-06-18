#include <bits/stdc++.h>
#include "ndarray.cpp"
#include <vector>
#include <typeinfo>
#include <random>
#include <chrono>
#pragma once
using namespace std;

// data type
static set<string> int_dtypes = {"i","l","x"};
static set<string> real_dtypes = {"f","d","e"};

class numcpp{
public:
    // all zero matrix
    template <typename T>
    ndarray<T> zeros(vector<int>& shape);

    // all one matrix
    template <typename T>
    ndarray<T> ones(vector<int>& shape);

    // equidistant array
    template <typename T>
    ndarray<T> arange(long long start, long long end);
    template <typename T>
    ndarray<T> linspace(double start, double end, long long N);

    // uniformly distributed random matrix
    template <typename T>
    ndarray<T> rand(double low, double high, vector<int>& shape);

    // normal distributed random matrix
    template <typename T>
    ndarray<T> normal(double mean, double scale, vector<int>& shape);
    // standard normal distributed random matrix
    template <typename T> 
    ndarray<T> randn(vector<int>& shape);
};


// all zero matrix
template <typename T>
ndarray<T> numcpp::zeros(vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    T arr[size] = 0;
    ndarray<T> mat(arr,shape);
    return mat;
}

// all one matrix
template <typename T>
ndarray<T> numcpp::ones(vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    T arr[size] = {1};
    ndarray<T> mat(arr,shape);
    return mat;
}

// equidistant array
template <typename T>
ndarray<T> numcpp::arange(long long start, long long end){
    long long size = end - start;
    T* arr = new T[size];
    for(long long i=0;i<size;++i) arr[i] = (T)(start + i);
    vector<int> shape = {(int)size};
    ndarray<T> mat(arr,shape);
    return mat;
}
// equidistant array
template <typename T>
ndarray<T> numcpp::linspace(double start, double end, long long N){
    long long size = N;
    T sep = (end - start) / (N-1);
    T arr[size];
    for(long long i=0;i<size;++i) arr[i] = (T)(start + i*sep);
    vector<int> shape = {(int)size};
    ndarray<T> mat(arr,shape);
    return mat;
}

// uniformly distributed random matrix
template <typename T>
ndarray<T> numcpp::rand(double low, double high, vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    // define data type
    typedef T value_type; 
    value_type _dtype;
    const type_info &dataInfo = typeid(_dtype);

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the int generator
    uniform_int_distribution<long> distribution_int(low,high);
    // the float generator
    uniform_real_distribution<double> distribution_real(low,high);

    // create array
    T arr[size];
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

// normal distributed random matrix
template <typename T>
ndarray<T> numcpp::normal(double mean, double scale, vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    normal_distribution<T> distribution(mean,scale);

    T arr[size];
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<T> mat(arr,shape);
    return mat;
}
// standart normal distributed random matrix
template <typename T>
ndarray<T> numcpp::randn(vector<int>& shape){
    return this->normal<T>(0,1,shape);
}