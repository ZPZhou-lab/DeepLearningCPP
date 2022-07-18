#include <bits/stdc++.h>
#include "ndarray.cpp"
#include <cstdlib>
#include <vector>
#include <typeinfo>
#include <random>
#include <chrono>
#pragma once
using namespace std;

// class for generate random numbers from various distributions
class randomBses{
public:
    // utility fuctions
    // randomly permute [0,1,2,...,x-1]
    ndarray<long long> permutation(long long x);
    // make a copy and shuffle the elements randomly
    template<typename _Tp>
    ndarray<_Tp> permutation(ndarray<_Tp> &x);
    // randomly permute a sequence in place
    template<typename _Tp>
    void shuffle(ndarray<_Tp> &array);
    // random sanple from 1-D array
    template<typename _Tp>
    ndarray<_Tp> choice(ndarray<_Tp> &array, vector<int>& shape, bool replace=true, vector<double> p=vector<double>());


    // Univariate distributions
    // uniformly distributed random matrix
    ndarray<double> uniform(double low, double high, vector<int> shape=vector<int>());
    // uniformly distributed random matrix for elements in [0,1)
    template <typename ...Args>
    ndarray<double> rand(Args...args);
    // Return random integers from the "discrete uniform" distribution
    template <typename _Tp>
    ndarray<_Tp> randint(long long low, long long high, vector<int> shape=vector<int>());

    // normal distributed random matrix
    ndarray<double> normal(double mean, double scale, vector<int> shape=vector<int>());
    // standard normal distributed random matrix
    template<typename ...Args>
    ndarray<double> randn(Args...args);

    // beta distribution over ``[0, 1]``
    ndarray<double> beta(double a, double b, vector<int> shape=vector<int>());
    // binomial distribution
    ndarray<double> binomial(int n, double p, vector<int> shape=vector<int>());
    // chisquare distribution
    ndarray<double> chisquare(double df, vector<int> shape=vector<int>());
    // exponential distribution
    ndarray<double> exponontial(double scale, vector<int> shape=vector<int>());
    // F(Fisher0Snedecor) distribution
    ndarray<double> f(double dfnum, double dfden, vector<int> shape=vector<int>());
    // Gamma distribution
    ndarray<double> gamma(double shape, double scale, vector<int> _shape=vector<int>());
    // Geometric distribution
    ndarray<double> geometric(double p, vector<int> shape=vector<int>());
    // Poisson distribution
    ndarray<double> poisson(double lam, vector<int> shape=vector<int>());
    
    // Multivariate distributions
    ndarray<double> multivariate_normal(ndarray<double>& mean, ndarray<double>& cov, vector<int> shape=vector<int>());

    // Standard distribution
    ndarray<double> standard_cauchy(vector<int> shape=vector<int>());
    ndarray<double> standard_t(double df, vector<int> shape=vector<int>());

};

// class for linear algebra method
class linaig{
public:
    // Cholesky decomposition
    template <typename _Tp>
    ndarray<double> cholesky(ndarray<_Tp> &array);

    // Compute the determinant of an array
    template <typename _Tp>
    double det(ndarray<_Tp> &array);

    // Compute the eigenvalues and right eigenvectors of a square array
    template <typename _Tp>
    pair<ndarray<double>, ndarray<double>> eig(ndarray<_Tp> &array);

    // Compute the (multiplicative) inverse of a matrix
    template <typename _Tp>
    ndarray<double> inv(ndarray<_Tp> &array);

    // Raise a square matrix to the (integer) power `n`
    template <typename _Tp>
    ndarray<double> matrix_power(ndarray<_Tp> &array, int n);

    // Matrix or vector norm
    template <typename _Tp>
    double norm(ndarray<_Tp> &array, double ord, int axis, bool keepdims=false);

    // Compute the qr factorization of a matrix
    template <typename _Tp>
    pair<ndarray<double>, ndarray<double>> qr(ndarray<_Tp> &array, string mode);

    // Solve a linear matrix equation, or system of linear scalar equations
    template <typename _Tp>
    ndarray<double> solve(ndarray<_Tp> &A, ndarray<_Tp> &b);

};

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

    // class for generate random numbers
    randomBses random;

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
ndarray<double> randomBses::uniform(double low, double high, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    uniform_real_distribution<double> distribution_real(low,high);

    // create array
    vector<double> arr(size);

    // generate random numbers
    auto dice = bind(distribution_real,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();

    // create ndarray
    ndarray<double> mat(arr,shape);
    return mat;
}

// Create an array of the given shape and populate it with random samples from a uniform distribution
template <typename ...Args>
ndarray<double> randomBses::rand(Args...args){
    vector<int> shape;
    shape = fetchArgs(shape,args...);
    return this->uniform(0, 1, shape);
}

/* 
Return random integers from the "discrete uniform" distribution of 
the specified dtype in the "closed" interval [`low`, `high`]
*/
template <typename _Tp>
ndarray<_Tp> randomBses::randint(long long low, long long high, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the int generator
    uniform_int_distribution<_Tp> distribution_int(low,high);

    // create array
    vector<_Tp> arr(size);

    // generate random numbers
    auto dice = bind(distribution_int,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();

    // create ndarray
    ndarray<_Tp> mat(arr,shape);
    return mat;
}


// normal distributed random matrix
ndarray<double> randomBses::normal(double mean, double scale, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    normal_distribution<double> distribution(mean,scale);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}
// standart normal distributed random matrix
template <typename ...Args>
ndarray<double> randomBses::randn(Args...args){
    vector<int> shape;
    shape = fetchArgs(shape,args...);
    return this->normal(0,1,shape);
}

// beta distribution over ``[0, 1]``
ndarray<double> randomBses::binomial(int n, double p, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    binomial_distribution<int> distribution(n,p);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// chisquare distribution
ndarray<double> randomBses::chisquare(double df, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    chi_squared_distribution<double> distribution(df);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// exponential distribution
ndarray<double> randomBses::exponontial(double scale, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    exponential_distribution<double> distribution(1 / scale);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// F(Fisher0Snedecor) distribution
ndarray<double> randomBses::f(double dfnum, double dfden, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    fisher_f_distribution<double> distribution(dfnum,dfden);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// Gamma distribution
ndarray<double> randomBses::gamma(double shape, double scale, vector<int> _shape){
    long long size = 1;
    for(auto s : _shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    gamma_distribution<double> distribution(shape,scale);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,_shape);
    return mat;
}

// geometric distribution
ndarray<double> randomBses::geometric(double p, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    geometric_distribution<int> distribution(p);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// Poisson distribution
ndarray<double> randomBses::poisson(double lam, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    poisson_distribution<int> distribution(lam);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// standard t(student) distribution
ndarray<double> randomBses::standard_t(double df, vector<int> shape){
    long long size = 1;
    for(auto s : shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    student_t_distribution<double> distribution(df);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,shape);
    return mat;
}

// standard cauchy distribution
ndarray<double> randomBses::standard_cauchy(vector<int> _shape){
    long long size = 1;
    for(auto s : _shape) size *= s;

    // create random number generator
    // seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    // the float generator
    cauchy_distribution<double> distribution(0,1);
    
    vector<double> arr(size);
    auto dice = bind(distribution,generator);
    for(long long i=0;i<size;++i) arr[i] = dice();
    ndarray<double> mat(arr,_shape);
    return mat;
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