#include <bits/stdc++.h>
#include <bits/types/clock_t.h>
#include <cassert>
#include <climits>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <string>
#include <cmath>
#include <sys/cdefs.h>
#include <unordered_set>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <stdarg.h>

using namespace std;

template <typename _Tp>
class ndarray;

// time conuter
clock_t tic, toc;

// get parameters from an indefinite length parameter list
template <typename _Tp>
vector<_Tp> fetchArgs(vector<_Tp>& fetch, _Tp arg){
    fetch.emplace_back(arg);
    return fetch;
}
template <typename _Tp, typename ...Args>
vector<_Tp> fetchArgs(vector<_Tp>& fetch, _Tp arg, Args...args){
    fetch.emplace_back(arg);
    return fetchArgs(fetch,args...);
}

// permutation of axes
vector<int> permute(vector<int>& vec, vector<int>& axes){
    // temporary variable
    vector<int> permuted(vec.size());
    for(int i=0;i<vec.size();++i) permuted[i] = vec[axes[i]];

    return permuted;
}

// fetch reduction size
long long __reduce_size(const vector<int> &shape, int ndim, vector<int> &axis){
    // init
    long long __size = 1;
    // axis set
    set<int> axis_set;
    for(auto j:axis) axis_set.insert(j);

    // compute new size
    for(int i=0;i<ndim;++i){
        if(axis_set.find(i) == axis_set.end()){
            __size *= shape[i];
        }
    }

    return __size;
}

long long __reduce_size(const vector<int> &shape, int ndim, int axis){
    // init
    long long __size = 1;

    // compute new size
    for(int i=0;i<ndim;++i){
        if(i != axis){
            __size *= shape[i];
        }
    }

    return __size;
}

// fetch reduction shape
vector<int> __reduce_shape(const vector<int> &shape, int ndim, vector<int> &axis){
    // init
    vector<int> __shape;
    // axis set
    set<int> axis_set;
    for(auto j:axis) axis_set.insert(j);

    // compute new shape
    for(int i=0;i<ndim;++i){
        if(axis_set.find(i) == axis_set.end()){
            __shape.emplace_back(shape[i]);
        }
    }

    return __shape;
}

vector<int> __reduce_shape(const vector<int> &shape, int ndim, int axis){
    // init
    vector<int> __shape;

    // compute new shape
    for(int i=0;i<ndim;++i){
        if(i != axis){
            __shape.emplace_back(shape[i]);
        }
    }

    return __shape;
}

// fetch reduction step
long long __reduce_step(const vector<int> &shape, int ndim, vector<int> &axis){
    // init
    long long step = 1;
    // axis set
    set<int> axis_set;
    for(auto j:axis) axis_set.insert(j);

    // compute new size
    for(int i=0;i<ndim;++i){
        if(axis_set.find(i) != axis_set.end()){
            step *= shape[i];
        }
    }

    return step;
}

// reduction help function for sum()
template <typename _Tp, typename T1>
void sum_reduction(T1 &a, _Tp &b){
    a += b;
}

// reduction help function for max()
template <typename _Tp, typename T1>
void max_reduction(T1 &a, _Tp &b){
    a = a < b ? b:a;
}

// reduction help function for min()
template <typename _Tp, typename T1>
void min_reduction(T1 &a, _Tp &b){
    a = a > b ? b:a;
}

// reduction help function for any()
template <typename _Tp, typename T1>
void any_reduction(T1 &a, _Tp &b){
    a = a || b;
    //a = (a == 1 || b != 0) ? 1:0;
}

// reduction help function for all()
template <typename _Tp, typename T1>
void all_reduction(T1 &a, _Tp &b){
    a = a && b;
    //a = (a == 1 && b != 1) ? 1:0;
}

// reduction help function for argmax()
template <typename _Tp>
bool argmax_reduction(_Tp &a, _Tp &b){
    return a > b ? true:false;
}

// reduction help function for argmin()
template <typename _Tp>
bool argmin_reduction(_Tp &a, _Tp &b){
    return a < b ? true:false;
}

// quicksort method
template <typename _Tp>
void quicksort(vector<_Tp> &array, long long left, long long right){
    long long low = left, high = right;
    _Tp tmp;
    if(low < high){
        tmp = array[low];
        while(low < high){
            while(high > low && array[high] >= tmp) --high;
            array[low] = array[high];
            while(low < high && array[low] <= tmp) ++low;
            array[high] = array[low];
        }
        array[low] = tmp;
        // recur to sort
        quicksort(array, left, low-1);
        quicksort(array, low+1, right);
    }
}

// quicksort with index
template <typename _Tp>
void quicksort(ndarray<_Tp> &array, vector<long long> &idx, long long left, long long right){
    long long low = left, high = right;
    _Tp tmp;
    long long tmp_idx;
    if(low < high){
        tmp = array[low];
        tmp_idx = idx[low];
        while(low < high){
            while(high > low && array[high] >= tmp) --high;
            array[low] = array[high];
            idx[low] = idx[high];
            while(low < high && array[low] <= tmp) ++low;
            array[high] = array[low];
            idx[high] = idx[low];
        }
        array[low] = tmp;
        idx[low] = tmp_idx;
        // recur to sort
        quicksort(array, idx, left, low-1);
        quicksort(array, idx, low+1, right);
    }
}

// inner product, sum of product
template <typename _Tp1, typename _Tp2>
double __inner_product(const vector<_Tp1> &arr1, const long long s1, const vector<_Tp2> &arr2, const long long s2, const long long n){
    // init result
    double s = 0;
    // compute inner product
    for(long long i=0;i<n;++i) s += arr1[s1+i] * arr2[s2+i];

    return s;
}

// method for compute matrix production
template <typename _Tp1, typename _Tp2>
vector<double> __matrix_prod(const vector<_Tp1> &mat1, const vector<_Tp2> &mat2, long long step){
    // compute number of rows and cols
    int m = mat1.size() / step, n = mat2.size() / step;
    
    // init result
    vector<double> prod(m*n,0);
    // init index
    long long idx = 0;
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            // compute inner product
            prod[idx] = __inner_product(mat1, i*step, mat2, j*step, step);
            idx++;
        }
    }

    return prod;
}
