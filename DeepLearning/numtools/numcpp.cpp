#include <bits/stdc++.h>
#include "ndarray.cpp"
#include <vector>
#include <typeinfo>
#include <random>
#include <chrono>
#pragma once
using namespace std;

class numcpp{
public:
    // all zero matrix
    template <typename _Tp>
    ndarray<_Tp> zeros(vector<int>& shape);

    // all one matrix
    template <typename _Tp>
    ndarray<_Tp> ones(vector<int>& shape);

    // equidistant array
    template <typename _Tp>
    ndarray<_Tp> arange(long long start, long long end);
    template <typename _Tp>
    ndarray<_Tp> linspace(double start, double end, long long N);

    // uniformly distributed random matrix
    template <typename _Tp>
    ndarray<_Tp> rand(double low, double high, vector<int>& shape);

    // normal distributed random matrix
    template <typename _Tp>
    ndarray<_Tp> normal(double mean, double scale, vector<int>& shape);
    // standard normal distributed random matrix
    template <typename _Tp> 
    ndarray<_Tp> randn(vector<int>& shape);


    // method reshape()
    template <typename _Tp>
    ndarray<_Tp> reshape(ndarray<_Tp> &array, vector<int> &shape);
    // method transpose()
    template<typename _Tp>
    ndarray<_Tp> transpose(ndarray<_Tp> &array, vector<int> &axes);
    // method flatten()
    template<typename _Tp>
    ndarray<_Tp> flatten(ndarray<_Tp> &array);
    // method expand_dims()
    template<typename _Tp>
    ndarray<_Tp> expand_dims(ndarray<_Tp> &array, vector<int> axis);

    // computation
    // method dot() for array
    template<typename _Tp1, typename _Tp2>
    ndarray<double> dot(ndarray<_Tp1> &arr1, ndarray<_Tp2> &arr2);

};


// all zero matrix
template <typename _Tp>
ndarray<_Tp> numcpp::zeros(vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    vector<_Tp> arr(size,0);
    ndarray<_Tp> mat(arr,shape);
    return mat;
}

// all one matrix
template <typename _Tp>
ndarray<_Tp> numcpp::ones(vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    vector<_Tp> arr(size,1);
    ndarray<_Tp> mat(arr,shape);
    return mat;
}

// equidistant array
template <typename _Tp>
ndarray<_Tp> numcpp::arange(long long start, long long end){
    long long size = end - start;
    vector<_Tp> arr(size,1);
    for(long long i=0;i<size;++i) arr[i] = (_Tp)(start + i);
    vector<int> shape = {(int)size};
    ndarray<_Tp> mat(arr,shape);
    return mat;
}
// equidistant array
template <typename _Tp>
ndarray<_Tp> numcpp::linspace(double start, double end, long long N){
    long long size = N;
    _Tp sep = (end - start) / (N-1);
    vector<_Tp> arr(size,1);
    for(long long i=0;i<size;++i) arr[i] = (_Tp)(start + i*sep);
    vector<int> shape = {(int)size};
    ndarray<_Tp> mat(arr,shape);
    return mat;
}

// uniformly distributed random matrix
template <typename _Tp>
ndarray<_Tp> numcpp::rand(double low, double high, vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;
    // define data type
    typedef _Tp value_type; 
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
    vector<_Tp> arr(size);
    if(int_dtypes.find(dataInfo.name()) != int_dtypes.end()){
        auto dice = bind(distribution_int,generator);
        for(long long i=0;i<size;++i) arr[i] = dice();
    }else{
        auto dice = bind(distribution_real,generator);
        for(long long i=0;i<size;++i) arr[i] = dice();
    }
    ndarray<_Tp> mat(arr,shape);
    return mat;
}

// normal distributed random matrix
template <typename _Tp>
ndarray<_Tp> numcpp::normal(double mean, double scale, vector<int>& shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    normal_distribution<_Tp> distribution(mean,scale);

    vector<_Tp> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<_Tp> mat(arr,shape);
    return mat;
}
// standart normal distributed random matrix
template <typename _Tp>
ndarray<_Tp> numcpp::randn(vector<int>& shape){
    return this->normal<_Tp>(0,1,shape);
}

// reshape
template <typename _Tp>
ndarray<_Tp> numcpp::reshape(ndarray<_Tp> &array, vector<int> &shape){
    return array.reshape(shape);
}

// reshape
template <typename _Tp>
ndarray<_Tp> numcpp::transpose(ndarray<_Tp> &array, vector<int> &axes){
    return array.transpose(axes);
}

// reshape
template <typename _Tp>
ndarray<_Tp> numcpp::flatten(ndarray<_Tp> &array){
    return array.flatten();
}

// dot
template <typename _Tp1, typename _Tp2>
ndarray<double> numcpp::dot(ndarray<_Tp1> &arr1, ndarray<_Tp2> &arr2){
    return arr1.dot(arr2);
}

// expand_dims
template <typename _Tp>
ndarray<_Tp> numcpp::expand_dims(ndarray<_Tp> &array, vector<int> axis){
    return array.expand_dims(axis);
}