#include <algorithm>
#include <bits/stdc++.h>
#include "ndarray.cpp"
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <regex>
#include <utility>
#include <vector>
#include <typeinfo>
#include <random>
#include <chrono>
#include <cmath>
#pragma once
using namespace std;

class numcpp{
private:
    // general method for mathematical map
    template<typename _Tp>
    ndarray<double> static _general_math_map(ndarray<_Tp> &array, _Tp (*func)(_Tp a));

public:
    // all zero matrix
    template <typename _Tp>
    ndarray<_Tp> static zeros(vector<int>& shape);

    // all one matrix
    template <typename _Tp>
    ndarray<_Tp> static ones(vector<int>& shape);

    // identical matrix
    template <typename _Tp>
    ndarray<_Tp> static eye(long long m,long long n=-1,long long diag=0);

    // equidistant array
    template <typename _Tp>
    ndarray<_Tp> static arange(long long start, long long end);
    template <typename _Tp>
    ndarray<_Tp> static linspace(double start, double end, long long N);

    // math computation
    template <typename _Tp>
    ndarray<double> static exp(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static log(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static sin(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static cos(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static tan(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static sinh(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static cosh(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static tanh(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static abs(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<double> static pow(ndarray<_Tp> &array, double r);

    // statistics method
    template <typename _Tp>
    ndarray<_Tp> static sum(ndarray<_Tp> &array, vector<int> axis, bool keepdim=false);
    template <typename _Tp>
    _Tp static sum(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<_Tp> static max(ndarray<_Tp> &array, vector<int> axis, bool keepdim=false);
    template <typename _Tp>
    _Tp static max(ndarray<_Tp> &array);

    template <typename _Tp>
    ndarray<_Tp> static min(ndarray<_Tp> &array, vector<int> axis, bool keepdim=false);
    template <typename _Tp>
    _Tp static min(ndarray<_Tp> &array);
    
    template <typename _Tp>
    ndarray<_Tp> static mean(ndarray<_Tp> &array, vector<int> axis, bool keepdim=false);
    template <typename _Tp>
    _Tp static mean(ndarray<_Tp> &array);

    // class for generate random numbers from various distributions
    class randomBase{
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
    // class for generate random numbers
    randomBase random;

    // class for linear algebra method
    class linaigBase{
    private:
        // LU solver to solve linear matrix equation
        template<typename _Tp>
        ndarray<double> static __LU_solver(ndarray<double> &L, ndarray<double> &U, ndarray<_Tp> &b);

        // determine whether the matrix is a positive definite matrix
        template<typename _Tp>
        bool static __check_positive_definite_mat(ndarray<_Tp> &array);
    
    public:
        // Cholesky decomposition
        template <typename _Tp>
        ndarray<double> static cholesky(ndarray<_Tp> &array);

        // Compute the determinant of an array
        template <typename _Tp>
        double static det(ndarray<_Tp> &array);

        // Compute the eigenvalues and right eigenvectors of a square array
        template <typename _Tp>
        pair<ndarray<double>, ndarray<double>> static eig(ndarray<_Tp> &array);

        // Compute the eigenvalues, 
          // Main difference between `eigvals` and `eig`: the eigenvectors aren't returned
        template <typename _Tp>
        ndarray<double> static eigvals(ndarray<_Tp> &array);

        // Compute the (multiplicative) inverse of a matrix
        template <typename _Tp>
        ndarray<double> static inv(ndarray<_Tp> &array);

        // Raise a square matrix to the (integer) power `n`
        template <typename _Tp>
        ndarray<double> static matrix_power(ndarray<_Tp> &array, int n);

        // Matrix or vector norm
        template <typename _Tp>
        ndarray<double> static norm(ndarray<_Tp> &array, int axis, double ord=2,bool keepdims=false);
        template <typename _Tp>
        double static norm(ndarray<_Tp> &array, double ord=2);
        // 1-norm, 2-norm, F-norm, inf-norm
        template <typename _Tp>
        double static norm(ndarray<_Tp> &array, string ord="2-norm");

        // Compute the QR factorization of a matrix
        template <typename _Tp>
        pair<ndarray<double>, ndarray<double>> static QR(ndarray<_Tp> &array, string mode);
    
        // Compute the LU factorization of a matrix
        template <typename _Tp>
        pair<ndarray<double>, ndarray<double>> static LU(ndarray<_Tp> &array);
    
        // Compute the SVD factorization of a matrix
        template <typename _Tp>
        pair<ndarray<double>, ndarray<double>> static SVD(ndarray<_Tp> &array, string mode);

        // Solve a linear matrix equation, or system of linear scalar equations
        template <typename _Tp>
        ndarray<double> static solve(ndarray<_Tp> &A, ndarray<_Tp> &b);

    };

    linaigBase linaig;

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

// identical matirx
template <typename _Tp>
ndarray<_Tp> numcpp::eye(long long m,long long n,long long diag){
    if(n == -1) n = m;
    
    // create a all zero matrix
    vector<int> __shape = {(int)m,(int)n};
    auto E = numcpp::zeros<double>(__shape);

    // assign elements
    long long pos = 0;
    while(pos + diag < m && pos + diag < n){
        E[pos * n + pos+diag] = 1;
        pos++;
    }

    return E;
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

// randomly permute [0,1,2,...,x-1]
ndarray<long long> numcpp::randomBase::permutation(long long x){
    // generate sequence [0,1,2,...,x-1]
    ndarray<long long> seq = arange<long long>(0,x);
    // set random seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    // do shuffle
    for(long long i=0;i<x;++i){
        // generate random number in interval [i,n]
        long long r_idx = std::rand() % (x - i) + i;

        // swap
        long long tmp = seq[i];
        seq[i] = seq[r_idx];
        seq[r_idx] = tmp;
    }

    return seq;
}

// make a copy and shuffle the elements randomly
template <typename _Tp>
ndarray<_Tp> numcpp::randomBase::permutation(ndarray<_Tp> &x){
    // get a copy
    auto permuted = x.flatten();

    // generate a permutation sequence
    auto seq = permutation(x.size());

    // assign elements
    for(long long i=0;i<x.size();++i) permuted[i] = x[seq[i]];

    return permuted;
}

// randomly permute a sequence in place
template <typename _Tp>
void numcpp::randomBase::shuffle(ndarray<_Tp> &array){
    // set random seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    // do shuffle
    for(long long i=0;i<array.size();++i){
        // generate random number in interval [i,n]
        long long r_idx = std::rand() % (array.size() - i) + i;

        // swap
        long long tmp = array[i];
        array[i] = array[r_idx];
        array[r_idx] = tmp;
    }

    return;
}

// random sample from 1-D array
template <typename _Tp>
ndarray<_Tp> numcpp::randomBase::choice(ndarray<_Tp> &array, vector<int>& shape, bool replace, vector<double> p){
    // check whether array is 1-D array
    __check_one_dimension(array.shape());

    // init sample array
    long long size = 1;
    for(auto s:shape) size *= s;
    vector<_Tp> arr(size,0);
    ndarray<_Tp> samples(arr,shape);

    // set random seed
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    srand(seed);

    if(replace){
        // sample with equal propobility
        if(p.size() == 0){
            for(long long i=0;i<size;++i){
                long long r_idx = std::rand() % array.size();
                samples[i] = array[r_idx];
            }
        }else{
        // sample according to propobility vector p

            // make sure all the elments(weights) in p are greater or equal to 0
            // and make sure the shape of propobility vecter matches array
            __check_propobility(p,array.size());;

            // compute propobility
            double p_sum = 0;
            for(auto pr : p) p_sum += pr;
            // create a vector to save propobility distribution
            vector<double> p_dist(p.size(),0);
            // init
            p_dist[0] = p[0] / p_sum;
            // compute propobility
            for(long long i=1;i<p.size();++i){
                p_dist[i] = p_dist[i-1] + p[i] / p_sum;
            }

            // generate random number in [0,1) and assign elements
            double r_num;
            long long r_idx;
            // seed
            default_random_engine generator(seed);
            // the float generator
            uniform_real_distribution<double> distribution_real(0,1);
            auto dice = bind(distribution_real,generator);

            for(long long i=0;i<size;++i){
                r_num = dice();
                // use binary search to find index
                r_idx = upper_bound(p_dist.begin(),p_dist.end(),r_num) - p_dist.begin();
                samples[i] = array[r_idx];
            }
        }
    }else{
        // make sure the size of array greater than the sample size
        __check_choice_sample(size,array.size());

        // create a vector to save elements
        vector<_Tp> elements(array.data());

        // generate propobility distribution
        if(p.size() == 0){
            // set same weight
            p = vector<double>(array.size(),1);
        }else{
            // check weight
            __check_propobility(p,array.size());;
        }

        // compute sum of weights
        double p_sum = 0;
        for(auto pr : p) p_sum += pr;

        for(long long i=0;i<size;++i){
            // generate a random number in [0,p_sum)
            double r_num = (double)std::rand() / INT_MAX * p_sum;
            // find location
            double r_sum = 0;
            long long r_idx = 0;

            while(r_idx < p.size()){
                r_sum += p[r_idx];
                if(r_sum >= r_num) break;
                r_idx++;
            }

            // assign elements
            samples[i] = array[r_idx];

            // update propobilit
            p_sum -= p[r_idx];
            p[r_idx] = 0;
        }
    }

    return samples;
}

// uniformly distributed random matrix
ndarray<double> numcpp::randomBase::uniform(double low, double high, vector<int> shape){
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
ndarray<double> numcpp::randomBase::rand(Args...args){
    vector<int> shape;
    shape = fetchArgs(shape,args...);
    return this->uniform(0, 1, shape);
}

/* 
Return random integers from the "discrete uniform" distribution of 
the specified dtype in the "closed" interval [`low`, `high`]
*/
template <typename _Tp>
ndarray<_Tp> numcpp::randomBase::randint(long long low, long long high, vector<int> shape){
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
ndarray<double> numcpp::randomBase::normal(double mean, double scale, vector<int> shape){
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
ndarray<double> numcpp::randomBase::randn(Args...args){
    vector<int> shape;
    shape = fetchArgs(shape,args...);
    return this->normal(0,1,shape);
}

// beta distribution over ``[0, 1]``
ndarray<double> numcpp::randomBase::binomial(int n, double p, vector<int> shape){
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
ndarray<double> numcpp::randomBase::chisquare(double df, vector<int> shape){
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
ndarray<double> numcpp::randomBase::exponontial(double scale, vector<int> shape){
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
ndarray<double> numcpp::randomBase::f(double dfnum, double dfden, vector<int> shape){
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
ndarray<double> numcpp::randomBase::gamma(double shape, double scale, vector<int> _shape){
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
ndarray<double> numcpp::randomBase::geometric(double p, vector<int> shape){
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
ndarray<double> numcpp::randomBase::poisson(double lam, vector<int> shape){
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
ndarray<double> numcpp::randomBase::standard_t(double df, vector<int> shape){
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
ndarray<double> numcpp::randomBase::standard_cauchy(vector<int> _shape){
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

// matrix power
template <typename _Tp>
ndarray<double> numcpp::linaigBase::matrix_power(ndarray<_Tp> &array, int n){
    // check the shape of ndarray
    __check_2darray(array.shape());
    __check_rows_equal_cols(array.shape());

    // create an identical matrix
    auto E = numcpp::eye<double>(array.shape()[0]);
    // copy array
    auto c_array = array;

    // compute matrix power in O(log(n))
    while(n){
        if(n&1){
            E = E.dot(c_array);
        }
        c_array = c_array.dot(c_array);
        n >>= 1;
    }

    return E;
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

// sum
template <typename _Tp>
ndarray<_Tp> numcpp::sum(ndarray<_Tp> &array, vector<int> axis, bool keepdim){
    return array.sum(axis,keepdim);
}
template <typename _Tp>
_Tp numcpp::sum(ndarray<_Tp> &array){
    return array.sum();
}

// max
template <typename _Tp>
ndarray<_Tp> numcpp::max(ndarray<_Tp> &array, vector<int> axis, bool keepdim){
    return array.max(axis,keepdim);
}
template <typename _Tp>
_Tp numcpp::max(ndarray<_Tp> &array){
    return array.max();
}

// min
template <typename _Tp>
ndarray<_Tp> numcpp::min(ndarray<_Tp> &array, vector<int> axis, bool keepdim){
    return array.min(axis,keepdim);
}
template <typename _Tp>
_Tp numcpp::min(ndarray<_Tp> &array){
    return array.min();
}

// mean
template <typename _Tp>
ndarray<_Tp> numcpp::mean(ndarray<_Tp> &array, vector<int> axis, bool keepdim){
    return array.mean(axis,keepdim);
}
template <typename _Tp>
_Tp numcpp::mean(ndarray<_Tp> &array){
    return array.mean();
}

// general method for mathematical map
template <typename _Tp>
ndarray<double> numcpp::_general_math_map(ndarray<_Tp> &array, _Tp (*func)(_Tp a)){
    auto trans = array.template astype<double>();
    for(long long i=0;i<array.size();++i) trans[i] = (double)func(trans[i]);

    return trans;
}

// exp
template <typename _Tp>
ndarray<double> numcpp::exp(ndarray<_Tp> &array){
    return _general_math_map(array,std::exp);
}

// log
template <typename _Tp>
ndarray<double> numcpp::log(ndarray<_Tp> &array){
    return _general_math_map(array,std::log);
}

// sin
template <typename _Tp>
ndarray<double> numcpp::sin(ndarray<_Tp> &array){
    return _general_math_map(array,std::sin);
}

// cos
template <typename _Tp>
ndarray<double> numcpp::cos(ndarray<_Tp> &array){
    return _general_math_map(array,std::cos);
}

// tan
template <typename _Tp>
ndarray<double> numcpp::tan(ndarray<_Tp> &array){
    return _general_math_map(array,std::tan);
}

// sinh
template <typename _Tp>
ndarray<double> numcpp::sinh(ndarray<_Tp> &array){
    return _general_math_map(array,std::sinh);
}

// cosh
template <typename _Tp>
ndarray<double> numcpp::cosh(ndarray<_Tp> &array){
    return _general_math_map(array,std::cosh);
}

// tanh
template <typename _Tp>
ndarray<double> numcpp::tanh(ndarray<_Tp> &array){
    return _general_math_map(array,std::tanh);
}

// abs
template <typename _Tp>
ndarray<double> numcpp::abs(ndarray<_Tp> &array){
    return _general_math_map(array,std::abs);
}

// abs
template <typename _Tp>
ndarray<double> numcpp::pow(ndarray<_Tp> &array, double r){
    auto trans = array.template astype<double>();
    for(long long i=0;i<array.size();++i) trans[i] = (double)std::pow(trans[i],r);

    return trans;
}

// Compute the eigenvalues and right eigenvectors of a square array
template <typename _Tp>
pair<ndarray<double>, ndarray<double>> numcpp::linaigBase::eig(ndarray<_Tp> &array){
    // check whether array is a square matrix
    __check_2darray(array.shape());
    __check_rows_equal_cols(array.shape());

    // change data type to double
    auto mat = array.template astype<double>();
    // the number of rows and cols for matrix
    int dim = mat.shape()[0];

    // init eigenvectors
    ndarray<double> eigenvecs = eye<double>(dim);
    // init eigenvalues
    vector<int> _shape = {dim};
    ndarray<double> eigenvals = zeros<double>(_shape);

    // iterations
    int iter = 0;
    // iteration precision
    double precision = 1e-6;
    // max iterations
    int maxIter = 100;

    // jacobi iteration to solve eigenvectors and eigenvalues
    while(true){

        // Remove diagonal elements 
        // and search for the largest element and subscript of the absolute value of the matrix
        double maxVal = mat(0,1);
        int m_row = 0, m_col = 1;
        for(int i=0;i<dim;++i){
            for(int j=0;j<dim;++j){
                double absVal = std::fabs(mat(i,j));
                if(i != j && absVal > maxVal){
                    maxVal = absVal;
                    m_row = i;
                    m_col = j;
                }
            }
        }

        // judge threshold condition
        if(maxVal < precision) break;
        if(iter > maxIter) break;
        // update iteration
        iter++;
        /* compute rotation matrix */
        // assign vertex
        double mat_xx = mat(m_row,m_row);
        double mat_xy = mat(m_row,m_col);
        double mat_yy = mat(m_col,m_col);

        double Angle = 0.5 * std::atan2(-2 * mat_xy, mat_yy - mat_xx);
        double sinA = std::sin(Angle), cosA = std::cos(Angle);
        double sin2A = std::sin(2*Angle), cos2A = std::cos(2*Angle);

        // update matrix
        mat(m_row,m_row) = mat_xx*cosA*cosA + mat_yy*sinA*sinA + 2*mat_xy*cosA*sinA;
        mat(m_col,m_col) = mat_xx*sinA*sinA + mat_yy*cosA*cosA - 2*mat_xy*cosA*sinA;
        mat(m_row,m_col) = 0.5*(mat_yy - mat_xx) * sin2A + mat_xy*cos2A;
        mat(m_col,m_row) = mat(m_row,m_col);

        for(int i=0;i<dim;++i){
            if(i != m_col && i != m_row){
                maxVal = mat(i,m_row);
                mat(i,m_row) = mat(i,m_col) * sinA + maxVal * cosA;
                mat(i,m_col) = mat(i,m_col) * cosA - maxVal * sinA;
            }
        }
        for(int j=0;j<dim;++j){
            if(j != m_col && j!= m_row){
                maxVal = mat(m_row,j);
                mat(m_row,j) = mat(m_col,j) * sinA + maxVal * cosA;
                mat(m_col,j) = mat(m_col,j) * cosA - maxVal * sinA;
            }
        }

        //compute eigenvector
        for(int i=0;i<dim;++i){
            maxVal = eigenvecs(i,m_row);
            eigenvecs(i,m_row) = eigenvecs(i,m_col) * sinA + maxVal * cosA;
            eigenvecs(i,m_col) = eigenvecs(i,m_col) * cosA - maxVal * sinA;
        }
    }

    // eigenvalue sorting
    std::map<double, int> eigenSort;
    for(int i=0;i<dim;++i){
        // assign eigenvals
        eigenvals[i] = mat(i,i);
        // assign index
        eigenSort.insert(make_pair(eigenvals[i], i));
    }

    // save eigenvector elements and adjust sign
    double *__tmpEigenVec = new double[dim*dim];
    // get the sorting pointer ptr and arrange the eigenvalues in reverse order
    std::map<double, int>::reverse_iterator ptr = eigenSort.rbegin();

    // assign elements from eigenvector to tmpEigenVec
    for(int j=0;j<dim;++ptr, ++j){
        // iter on the i-th row of the eigenvector
        for(int i=0;i<dim;++i){
            __tmpEigenVec[i*dim + j] = eigenvecs(i,ptr->second);
        }
        eigenvals[j] = ptr->first; // assign eigenvalue in order
    }

    for(int i = 0; i < dim; i++){
        // the sum of the i-th eigenvector
        double sumVec = 0;
        for(int j = 0; j < dim; j++)
            sumVec += __tmpEigenVec[j * dim + i];
        // If the sum of the eigenvectors is less than zero, 
        // the sign of the eigenvector is adjusted
        if(sumVec < 0) for(int j = 0; j < dim; j++) __tmpEigenVec[j * dim + i] *= -1;

    }
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++) eigenvecs(i,j) = __tmpEigenVec[i * dim + j];
    }

    // release memory
    delete [] __tmpEigenVec;

    return make_pair(eigenvals, eigenvecs);

}

// Compute the eigenvalues, 
template <typename _Tp>
ndarray<double> numcpp::linaigBase::eigvals(ndarray<_Tp> &array){
    auto eigen = numcpp::linaigBase::eig(array);
    return eigen.first;
}

// Compute the LU factorization of a matrix
template <typename _Tp>
pair<ndarray<double>, ndarray<double>> numcpp::linaigBase::LU(ndarray<_Tp> &array){
   // check whether array is a square matrix
    __check_2darray(array.shape());
    __check_rows_equal_cols(array.shape());

    // the dim of the array
    int dim = array.shape()[0];

    // init
    vector<int> _shape = {dim,dim};
    auto L = numcpp::zeros<double>(_shape), 
         U = numcpp::zeros<double>(_shape);

    // compute LU factorization
    for(int i=0;i<dim;++i){
        for(int j=i;j<dim;++j){
            /*
            U[i,j] = A[i,j] - sum(L[i,0:i]*U[0:i,j])
            L[j,i] = (A[j,i] - sum(L[j,0:i]*U[0:i,i])) / U[i,i]
            */
            double _tmp1 = 0, _tmp2 = 0;
            
            for(int k=0;k<i;++k) _tmp1 += L(i,k)*U(k,j);
            U(i,j) = array(i,j) - _tmp1;

            for(int k=0;k<i;++k) _tmp2 += L(j,k)*U(k,i);
            L(j,i) = (array(j,i) - _tmp2) / U(i,i);
        }
    }

    return make_pair(L, U);
}

//  Solve a linear matrix equation, or system of linear scalar equations
template <typename _Tp>
ndarray<double> numcpp::linaigBase::solve(ndarray<_Tp> &A, ndarray<_Tp> &b){
    // check shape
    __check_one_dimension(b.shape());
    __check_rows_equal(A.shape()[0],b.size());
    
    // do LU factorization
    auto lu = numcpp::linaigBase::LU(A);
    auto L = lu.first, U = lu.second;

    return __LU_solver(L, U, b);
}

// LU solver to solve linear matrix equation
template<typename _Tp>
ndarray<double> numcpp::linaigBase::__LU_solver(ndarray<double> &L, ndarray<double> &U, ndarray<_Tp> &b){
    /*
    Ax = b -> (LU)x = b -> L(Ux) = b -> Ly = b -> Ux = y
    */
    int dim = b.size();

    vector<int> __shape = {dim,1};
    ndarray<double> x = numcpp::zeros<double>(__shape),
                    y = numcpp::zeros<double>(__shape);
    
    double _tmp ;
    // solve Ly = b
    for(int i=0;i<dim;++i){
        _tmp = 0;
        for(int j=0;j<i;++j){
            _tmp += L(i,j)*y[j];
        }
        y[i] = b[i] - _tmp;
    }

    // solve Ux = y
    for(int i=dim-1;i>-1;--i){
        _tmp = 0;
        for(int j=i+1;j<dim;++j){
            _tmp += U(i,j)*x[j];
        }
        x[i] = (y[i] - _tmp) / U(i,i);
    }

    return x;
}

// Compute the (multiplicative) inverse of a matrix
template <typename _Tp>
ndarray<double>  numcpp::linaigBase::inv(ndarray<_Tp> &array){
    // do LU factorization
    auto lu = numcpp::linaigBase::LU(array);
    auto L = lu.first, 
         U = lu.second;

    int dim = array.shape()[0];
    // create an identical vector
    vector<int> __shape = {dim};
    ndarray<double> e = numcpp::zeros<double>(__shape);
    e[0] = 1.0;

    // init inv(array)
    __shape = array.shape();
    auto inv_array = numcpp::zeros<double>(__shape);

    // solve the first column
    auto inv_col = __LU_solver(L, U, e);
    for(int i=0;i<dim;++i) inv_array(i,0) = inv_col[i];

    // sovle other columns
    for(int j=1;j<dim;++j){
        // update vector e
        e[j-1] = 0.0;
        e[j] = 1.0;

        // solve the j-th column
        inv_col = __LU_solver(L, U, e);
        for(int i=0;i<dim;++i) inv_array(i,j) = inv_col[i];
    }

    return inv_array;
}

// Compute the determinant of an array
template <typename _Tp>
double numcpp::linaigBase::det(ndarray<_Tp> &array){
    // compute eigenvalues
    auto eigvals = numcpp::linaigBase::eigvals(array);

    // compute det
    double _det = 1.0;
    for(long long i=0;i<eigvals.size();++i) _det *= eigvals[i];

    return _det;
}

// determine whether the matrix is a positive definite matrix
template<typename _Tp>
bool numcpp::linaigBase::__check_positive_definite_mat(ndarray<_Tp> &array){
    // fetch eigenvalues
    auto eigenvalues = numcpp::linaigBase::eigvals(array);

    // all eigenvalue should greater than 0
    for(long long i=0;i<eigenvalues.size();++i){
        if(eigenvalues[i] <= 0) return false;
    }

    return true;
}
    

// Cholesky decomposition
template <typename _Tp>
ndarray<double> numcpp::linaigBase::cholesky(ndarray<_Tp> &array){
    // Determine whether the matrix is a positive definite matrix
    if(!numcpp::linaigBase::__check_positive_definite_mat(array)){
        printf("Matrix is not positive definite!\n");
        assert(false);
    }

    // do decomposition
    // init
    vector<int> __shape = array.shape();
    int dim = __shape[0];
    auto L = numcpp::zeros<double>(__shape);

    for(int i=0;i<dim;++i){
        double _tmp = 0;
        for(int j=0;j<i;++j) _tmp += L(i,j) * L(i,j);
        _tmp = array(i,i) - _tmp;

        // compute diag element
        L(i,i) = _tmp > 0 ? std::sqrt(_tmp) : 0;

        // compute the i-th elements
        for(int j=i+1;j<dim;++j){
            _tmp = 0;
            for(int k=0;k<i;++k) _tmp += L(j,k) * L(i,k);

            L(j,i) = (array(j,i) - _tmp) / L(i,i);
        }
    }

    return L;
}

// Matrix or vector norm
template <typename _Tp>
double numcpp::linaigBase::norm(ndarray<_Tp> &array, string ord){
    // check 2-dimensional
    __check_2darray(array.shape());

    double _norm = 0;
    if(ord == "2-norm"){
        // compute eigvalues
        auto mat = array.T().dot(array);
        auto eigvalues = numcpp::linaigBase::eigvals(mat);

        // 2-norm equals to the square-root of the max eigvalue of A'A
        _norm = std::sqrt(eigvalues.max());
    
    }else if(ord == "1-norm"){
        // compute sum of columns
        vector<int> axis = {0};
        auto col_sum = numcpp::abs(array).sum(axis);

        // 1-norm equals to the max element of columns sum
        _norm = col_sum.max();
    
    }else if(ord == "f-norm"){
        // compute sum of the square of elements
        auto sum_square = numcpp::pow(array,2).sum();

        // Forbenius norm equals to the square root of sum of the square elements
        _norm = std::sqrt(sum_square);
    
    }else if(ord == "inf-norm"){
        // compute sum of rows
        vector<int> axis = {1};
        auto row_sum = numcpp::abs(array).sum(axis);

        // inf-norm equals to the max element of rows sum
        _norm = row_sum.max();
    
    }

    return _norm;
}

template <typename _Tp>
double numcpp::linaigBase::norm(ndarray<_Tp> &array, double ord){
    // check norm ord
    __check_norm_ord(array.shape());

    double _norm = 0;

    for(long long i=0;i<array.size();++i) _norm += std::pow(std::abs((double)array[i]),ord);

    _norm = std::pow(_norm,1 / ord);

    return _norm;
}

// Matrix or vector norm
template <typename _Tp>
ndarray<double> numcpp::linaigBase::norm(ndarray<_Tp> &array, int axis, double ord, bool keepdims){

}
