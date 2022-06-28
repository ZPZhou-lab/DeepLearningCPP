#include <bits/stdc++.h>
#include <bits/types/clock_t.h>
#include <cassert>
#include <climits>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <string>
#include <cmath>
#include <sys/cdefs.h>
#include <unordered_set>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <stdarg.h>

using namespace std;

template <typename T>
class ndarray;

// time conuter
clock_t tic, toc;

// fetch reduction size
long long __reduce_size(vector<int> &shape, int ndim, vector<int> &axis){
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

long long __reduce_size(vector<int> &shape, int ndim, int axis){
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
vector<int> __reduce_shape(vector<int> &shape, int ndim, vector<int> &axis){
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

vector<int> __reduce_shape(vector<int> &shape, int ndim, int axis){
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
long long __reduce_step(vector<int> &shape, int ndim, vector<int> &axis){
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
template <typename T, typename T1>
void sum_reduction(T1 &a, T &b){
    a += b;
}

// reduction help function for max()
template <typename T, typename T1>
void max_reduction(T1 &a, T &b){
    a = a < b ? b:a;
}

// reduction help function for min()
template <typename T, typename T1>
void min_reduction(T1 &a, T &b){
    a = a > b ? b:a;
}

// reduction help function for any()
template <typename T, typename T1>
void any_reduction(T1 &a, T &b){
    a = a || b;
    //a = (a == 1 || b != 0) ? 1:0;
}

// reduction help function for all()
template <typename T, typename T1>
void all_reduction(T1 &a, T &b){
    a = a && b;
    //a = (a == 1 && b != 1) ? 1:0;
}

// reduction help function for argmax()
template <typename T>
bool argmax_reduction(T &a, T &b){
    return a > b ? true:false;
}

// reduction help function for argmin()
template <typename T>
bool argmin_reduction(T &a, T &b){
    return a < b ? true:false;
}

// quicksort method
template <typename T>
void quicksort(vector<T> &array, long long left, long long right){
    long long low = left, high = right;
    T tmp;
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
template <typename T>
void quicksort(ndarray<T> &array, vector<long long> &idx, long long left, long long right){
    long long low = left, high = right;
    T tmp;
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