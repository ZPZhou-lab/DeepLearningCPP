#include <bits/stdc++.h>
#include <bits/types/clock_t.h>
#include <cassert>
#include <cfloat>
#include <climits>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <string>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <stdarg.h>
#include "check.cpp"
#include "utils.cpp"
#pragma once
using namespace std;

// data type
static set<string> int_dtypes = {"i","l","x"};
static set<string> real_dtypes = {"f","d","e"};

template <typename T>
class ndarray{
private:
    // define _data type
    typedef T value_type; 
    value_type _dtype;

    // store all elements in a vector
    vector<T> _data; 
    // the numver of elements
    long long _size;

    // the numver of the dimensions
    int _ndim; 
    // a vector represent how many elements stored in each dimension
    vector<int> _shape; 
    // a vector represents how many steps would across for each dimension
    vector<int> _strides; 

    // a vector represents the axes
    vector<int> _axes;

    // cumulative prod of shape, used to locate element
    vector<long long> __shape_cumprod;
    // maintenance function of shape_cumprod
    void __update_shape_cumprod(void);

    // internal(private) access method, check is omitted
    T __item(long long args);
    T __item(vector<int>& args);
    vector<int> __item_loc(long long args, vector<int> axis=vector<int>());

    // reuction method for sum(), max(), min()
    template<typename T1>
    ndarray<T1> _reduction(vector<int> axis, void (*func)(T1 &a,T &b), bool keepdim, vector<T1> &arr, 
                       long long step, vector<int> &__shape, long long __size);
    // reduction method for argmax(), argmin()
    ndarray<int> argreduction(int axis, bool (*func)(T &a, T &b));

public:
    // default constructer
    ndarray();
    // constructer
    ndarray(vector<T>& array, vector<int>& shape);
    ndarray(vector<T>& array, vector<int>& shape, vector<int>& strides, vector<int>& axes);
    // destructor
    ~ndarray();
    
    // access element
    T item(long long args); // access element by flat index
    T item(vector<int>& args); // access element by an nd-index into the array
    
    T at(long long idx); // access element by index

    // return _data type
    const string dtype(void);
    // return the number of elements
    const long long size(void);
    // return the number of dimensions
    const int ndim(void);
    // return the shape of the nd-array
    const vector<int> shape(void);
    // return the strides of the nd-array
    const vector<int> strides(void);
    // fetch _data
    const vector<T> &data(void);

    // array transform
    ndarray transpose(vector<int>& axes);
    ndarray reshape(vector<int>& shape);
    ndarray flatten(void);
    ndarray squeeze(vector<int> axis=vector<int>());

    // array operation
    ndarray sum(vector<int> axis, bool keepdim=false);
    T sum(void);
    ndarray max(vector<int> axis, bool keepdim=false);
    T max(void);
    ndarray min(vector<int> axis, bool keepdim=false);
    T min(void);
    ndarray<double> mean(vector<int> axis, bool keepdim=false);
    double mean(void);
    ndarray<int> any(vector<int> axis, bool keepdim=false);
    bool any(void);
    ndarray<int> all(vector<int> axis, bool keepdim=false);
    bool all(void);
    long long argmax(void);
    ndarray<int> argmax(int axis);
    long long argmin(void);
    ndarray<int> argmin(int axis);
    
    // show matrix
    void show(void);

    // opetator reload
    // operator reload []
    T &operator [] (long long args);
    const T &operator [] (long long args) const;

    // operation between ndarray and real number
    template<typename T1>
    ndarray operator - (const T1 b);
    template<typename T1>
    ndarray<double> operator * (const T1 b);
    template<typename T1>
    ndarray<double> operator / (const T1 b);
    template<typename T1>
    ndarray operator + (const T1 b);

    // operation between ndarray and ndarray
    template<typename T1>
    ndarray operator + (ndarray<T1> &b);
    template<typename T1>
    ndarray operator - (ndarray<T1> &b);
    template<typename T1>
    ndarray<double> operator * (ndarray<T1> &b);
    template<typename T1>
    ndarray<double> operator / (ndarray<T1> &b);

};

// get parameters from an indefinite length parameter list
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

// permutation of axes
vector<int> permute(vector<int>& vec, vector<int>& axes){
    // temporary variable
    vector<int> permuted(vec.size());
    for(int i=0;i<vec.size();++i) permuted[i] = vec[axes[i]];

    return permuted;
}

// default constructer
template <typename T>
ndarray<T>::ndarray(){
    this->_ndim = 0;
    this->_size = 0;
}

// constructer
template <typename T>
ndarray<T>::ndarray(vector<T>& array, vector<int>& shape){
    // compute size
    long long size = 1;
    for(auto s : shape) size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape((long long)array.size(),size);

    this->_data = array;
    this->_shape = shape;
    this->_ndim = shape.size();
    this->_size = size;

    // compute strides
    this->_strides = vector<int>(this->_ndim,1);
    for(int i=_ndim-1;i>0;i--) this->_strides[i-1] = this->_strides[i] * this->_shape[i];

    // add axes
    this->_axes = vector<int>(this->_ndim,0);
    for(int i=0;i<_ndim;++i) this->_axes[i] = i;

    __update_shape_cumprod();

}

// constructer
template <typename T>
ndarray<T>::ndarray(vector<T>& array, vector<int>& shape, vector<int>& strides, vector<int>& axes){
    // compute size
    long long size = 1;
    for(auto s:shape) size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape((long long)array.size(),size);

    this->_data = array;
    this->_shape = shape;
    this->_size = size;
    this->_ndim = shape.size();

    this->_strides = strides;
    this->_axes = axes;

    __update_shape_cumprod();
}

//  destructer
template <typename T>
ndarray<T>::~ndarray(){
    cout<<"Destructor called"<<endl;
}

// operator reload
template <typename T>
template <typename T1>
ndarray<double> ndarray<T>::operator/(const T1 b){
    vector<double> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] /= b;

    ndarray<double> trans(res,this->_shape);

    return trans;
}


template<typename T>
template<typename T1>
ndarray<T> ndarray<T>::operator+(const T1 b){
    vector<T> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] += b;

    ndarray<T> trans(res,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<double> ndarray<T>::operator*(const T1 b){
    vector<double> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] *= b;

    ndarray<double> trans(res,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator-(const T1 b){
    vector<T> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] -= b;

    ndarray<T> trans(res,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator+(ndarray<T1> &b){
    ndarray<T> flat1 = this->flatten();
    ndarray<T1> flat2 = b.flatten();
    vector<T> array(flat1.data());
    for(int i=0;i<this->_size;++i) array[i] += flat2.at(i);

    ndarray<T> trans(array,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<T> ndarray<T>::operator-(ndarray<T1> &b){
    ndarray<T> flat1 = this->flatten();
    ndarray<T1> flat2 = b.flatten();
    vector<T> array(flat1.data());
    for(int i=0;i<this->_size;++i) array[i] -= flat2.at(i);

    ndarray<T> trans(array,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<double> ndarray<T>::operator*(ndarray<T1> &b){
    ndarray<T> flat1 = this->flatten();
    ndarray<T1> flat2 = b.flatten();
    vector<double> array(flat1.data());
    for(int i=0;i<this->_size;++i) array[i] *= flat2.at(i);

    ndarray<double> trans(array,this->_shape);

    return trans;
}

template <typename T>
template <typename T1>
ndarray<double> ndarray<T>::operator/(ndarray<T1> &b){
    ndarray<T> flat1 = this->flatten();
    ndarray<T1> flat2 = b.flatten();
    vector<double> array(flat1.data());
    for(int i=0;i<this->_size;++i) array[i] /= flat2.at(i);

    ndarray<double> trans(array,this->_shape);

    return trans;
}


// maintenance function of shape_cumprod
template <typename T>
void ndarray<T>::__update_shape_cumprod(void){
    this->__shape_cumprod = vector<long long>(_ndim,1);
    for(int i=_ndim-1;i>0;i--) this->__shape_cumprod[i-1] = this->__shape_cumprod[i] * this->_shape[i];
}

// get flat index
inline long long __flat_idx(vector<int>& loc, vector<int>& strides){
    // initial flat index
    long long flat = 0;

    // compute flat index
    for(int i=0;i<loc.size();++i){
        flat += strides[i] * loc[i];
    }

    return flat;
}

// get item location vector
template <typename T>
inline vector<int> ndarray<T>::__item_loc(long long args, vector<int> axis){
    // initial lication
    vector<int> loc(_ndim,0);
    
    // compute location
    for(int i=0;i<_ndim;++i){
        loc[i] = (args / this->__shape_cumprod[i]);
        args %= this->__shape_cumprod[i];
    }
    
    for(auto j:axis) loc[j] = 0;

    return loc;
}

// access element by flat index
template <typename T>
inline T ndarray<T>::item(long long args){
    // check index
    __check_index(args,this->_size);

    // initial flat index
    long long flat = 0;    

    // compute flat index
    for(int i=0;i<_ndim;++i){
        flat += this->_strides[i] * (args / this->__shape_cumprod[i]);
        args %= this->__shape_cumprod[i];
    }
    return this->_data[flat];
}

// access element by nd-array index
template <typename T>
inline T ndarray<T>::item(vector<int>& args){
    // initial flat index
    long long flat = 0;

    // compute flat index
    for(int i=0;i<_ndim;++i){
        // check index
        __check_index(args[i],this->_shape[i],i);

        flat += this->_strides[i] * args[i];
    }

    return this->_data[flat];
}

// access element by seperate nd-array index
template <typename T>
inline T ndarray<T>::at(long long idx){
    // return element
    return this->_data[idx];
}

template<typename T>
T &ndarray<T>::operator[](long long args){
    return this->_data[args];
}

template <typename T>
const T &ndarray<T>::operator[](long long args) const{
    return this->_data[args];
}

// return _data type
template <typename T>
const string ndarray<T>::dtype(void){
    map<string,string> dtypes = {
        {"i","int"}, {"f","float"}, {"d","double"}, {"l","long"}, {"b","bool"},
        {"e", "long double"}, {"x","long long"}
    };
    const type_info &_dataInfo = typeid(this->_dtype);
    return dtypes[_dataInfo.name()];
}

// the number of elements
template <typename T>
const long long ndarray<T>::size(void){
    return this->_size;
}

// the number of dimensions
template <typename T>
const int ndarray<T>::ndim(void){
    return this->_ndim;
}

// the number of elements in each dimension
template <typename T>
const vector<int> ndarray<T>::shape(void){
    return this->_shape;
}

// the strides of the nd-array
template <typename T>
const vector<int> ndarray<T>::strides(void){
    return this->_strides;
}

// fetch data
template <typename T>
const vector<T> &ndarray<T>::data(void){
    return this->_data;
}

// print the array
template <typename T>
void ndarray<T>::show(void){
    const type_info &_dataInfo = typeid(this->_dtype);
    string dtype = _dataInfo.name();
    if(this->_ndim == 1){
        if(int_dtypes.find(dtype) == int_dtypes.end()){
            for(auto d : _data) printf("%12.4f", d);
        }else{
            for(auto d : _data) printf("%8lld", d);
        }
        printf("\n");
    }else{
        // compute cummulative prod of shape
        vector<long long> cumprod_shape(_ndim,1);
        for(int j=1;j<_ndim;++j) cumprod_shape[j] = cumprod_shape[j-1] * this->_shape[_ndim-j];

        // print each element
        for(long long i=0;i<this->_size;++i){
            if(int_dtypes.find(dtype) == int_dtypes.end()){
                printf("%12.4f", item(i));
            }else{
                printf("%8lld", item(i));
            }
            
            // wrap if crossing a dimension
            for(int j=1;j<this->_ndim;++j){
                if((i+1)%cumprod_shape[j] == 0) printf("\n");
            }
        }
    }
    printf("\n");
}

// transpose
template <typename T>
ndarray<T> ndarray<T>::transpose(vector<int>& axes){
    // anomaly detection
    __check_axes(axes, this->_ndim);

    // update strides
    vector<int> __strides = permute(this->_strides, axes);
    // update shape
    vector<int> __shape = permute(this->_shape, axes);
    // update axes
    vector<int> __axes = permute(this->_shape, axes);
    
    // transformed array
    ndarray<T> trans(this->_data,__shape,__strides,__axes);
    return trans;
}

// reshape
template <typename T>
ndarray<T> ndarray<T>::reshape(vector<int> &shape){
    // anomaly detection
    long long check_size = 1;
    for(auto s:shape) check_size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape(this->_size,check_size);

    // initialization
    ndarray<T> trans;

    // check whether the position of elements in the array needs to be adjusted
    bool flag = false;
    for(int i=0;i<_ndim;i++){
        if(this->_axes[i] != i){
            flag = true;
            break;
        }
    }

    // update ndim
    int __ndim = shape.size();

    // update strides
    vector<int> __strides = vector<int>(__ndim,1);
    for(int i=__ndim-1;i>0;i--) __strides[i-1] = __strides[i] * shape[i];

    // update axes
    vector<int> __axes = vector<int>(__ndim,0);
    for(int i=0;i<__ndim;++i) __axes[i] = i;

    // use permutation to adjust the position of elements then reshape
    if(flag){
        // adjust the position of elements
        // copy construction
        vector<T> array(this->_data);

        // element table switching
        for(long long i=0;i<this->_size;++i){
            array[i] = item(i);
        }

        trans = ndarray<T>(array,shape,__strides,__axes);
    }
    // do not need to adjust the position of elements
    else{
        trans = ndarray<T>(this->_data,shape,__strides,__axes);
    }

    return trans;
}

// flatten
template <typename T>
ndarray<T> ndarray<T>::flatten(void){
    vector<int> shape = {(int)this->_size};
    return reshape(shape);
}

// squeeze
template <typename T>
ndarray<T> ndarray<T>::squeeze(vector<int> axis){
    // initialization
    vector<int> __axes, __shape, __strides, __axes_idx;
    int __ndim = 0;

    // squeeze all dimensions whose shape equall to 1
    if(axis.size() == 0){
        for(int i=0;i<_ndim;++i){
            // squeeze the dimension when shape equals to 1
            if(this->_shape[i] != 1){
                // update shape and strides
                __shape.emplace_back(this->_shape[i]);
                __strides.emplace_back(this->_strides[i]);

                // update axes
                __axes.emplace_back(this->_axes[i]);
                // record axes index
                __axes_idx.emplace_back(__ndim);
                __ndim++;
            }
        }
    }
    // squeeze the specified axis
    else{
        // check squeeze
        __check_squeeze(axis,this->_shape);

        // squeeze the dimension specified
        for(int i=0;i<_ndim;++i){
            // determine whether squeeze the axis i
            bool flag = true;
            for(auto j:axis){
                if(i == j){
                    flag = false;
                    break;
                }
            }
            
            if(flag){
                // update shape and strides
                __shape.emplace_back(this->_shape[i]);
                __strides.emplace_back(this->_strides[i]);

                // update axes
                __axes.emplace_back(this->_axes[i]);
                // record axes index
                __axes_idx.emplace_back(__ndim);
                __ndim++;
            }
        }

    }

    // sort the axes and adjust axes index
    // insert-sort
    for(int j=1;j<__ndim;j++){
        for(int i=j;i>0;i--){
            if(__axes[i] < __axes[i-1]){
                // swap
                swap(__axes[i],__axes[i-1]);
                swap(__axes_idx[i],__axes_idx[i-1]);
            }else{
                break;
            }
        }
    }
    
    ndarray<T> trans(this->_data,__shape,__strides,__axes_idx);

    return trans;
}

// sum
template <typename T>
T ndarray<T>::sum(void){
    T s = 0;
    for(auto e:this->_data) s += e;

    return s;
}

// max
template <typename T>
T ndarray<T>::max(void){
    T s = INT_MIN;
    for(auto e:this->_data) s = std::max(s,e);

    return s;
}

// min
template <typename T>
T ndarray<T>::min(void){
    T s = INT_MAX;
    for(auto e:this->_data) s = std::min(s,e);

    return s;
}

// any()
template <typename T>
bool ndarray<T>::any(void){
    bool flag = false;
    for(auto e:this->_data) flag = flag || e;

    return flag;
}

// all()
template <typename T>
bool ndarray<T>::all(void){
    bool flag = true;
    for(auto e:this->_data) flag = flag && e;

    return flag;
}

template <typename T>
ndarray<T> ndarray<T>::sum(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<T> arr(__size,0);
    return this->_reduction(axis, sum_reduction, keepdim, arr, step, __shape, __size);
}

template <typename T>
ndarray<T> ndarray<T>::max(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<T> arr(__size,INT_MIN);
    return this->_reduction(axis, max_reduction, keepdim, arr, step, __shape, __size);
}

template <typename T>
ndarray<T> ndarray<T>::min(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<T> arr(__size,INT_MAX);
    return this->_reduction(axis, min_reduction, keepdim, arr, step, __shape, __size);
}

template <typename T>
ndarray<int> ndarray<T>::any(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<int> arr(__size,0);
    return this->_reduction(axis, any_reduction, keepdim, arr, step, __shape, __size);
}

template <typename T>
ndarray<int> ndarray<T>::all(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<int> arr(__size,1);
    return this->_reduction(axis, all_reduction, keepdim, arr, step, __shape, __size);
}

// reduction method for sum(), min(), max()
template <typename T>
template <typename T1>
ndarray<T1> ndarray<T>::_reduction(vector<int> axis, void (*func)(T1 &a, T &b), bool keepdim, vector<T1> &arr,
                                   long long step, vector<int> &__shape,long long __size){
    // initialization
    ndarray<T1> trans(arr,__shape);
    // adjust elements
    // get a copy of ndarray
    ndarray<T> copy(this->_data,this->_shape,this->_strides,this->_axes);
    // add dimensions
    vector<int> n_shape(this->_shape);
    for(auto j:axis){
        n_shape.emplace_back(1);
    }
    copy = copy.reshape(n_shape);

    // transpose axis
    vector<int> n_axes;
    for(int i=0;i<(this->_ndim + axis.size());++i) n_axes.emplace_back(i);
    for(int i=0;i<axis.size();++i) swap(n_axes[axis[i]],n_axes[_ndim+i]);
    copy = copy.transpose(n_axes);

    // delete the additional dimension
    copy = copy.squeeze();
    // flatten the array
    copy = copy.flatten();

    // assign elements
    for(long long i=0;i<__size;++i){
        for(long long j=0;j<step;++j){
            func(trans[i],copy[i*step+j]);
        }
    }

    return trans;
}

// mean
template <typename T>
double ndarray<T>::mean(void){
    double m;

    T s = this->sum();
    m = s / this->_size;

    return m;
}

// mean
template <typename T>
ndarray<double> ndarray<T>::mean(vector<int> axis, bool keepdim){
    // figure sum
    ndarray<T> s = this->sum(axis,keepdim);

    // cimpute number of elements
    long long __num = 1;
    for(auto i:axis) __num *= this->_shape[i];    

    // compute mean
    ndarray<double> trans = s / __num;

    return trans;
}

// argmax
template <typename T>
long long ndarray<T>::argmax(void){
    ndarray<T> flat = this->flatten();
    long long idx = 0;
    T maxVal = flat[0];
    for(long long i=0;i<this->_size;++i){
        if(flat[i] > maxVal){
            maxVal = flat[i];
            idx = i;
        }
    }

    return idx;
}

// argmin
template <typename T>
long long ndarray<T>::argmin(void){
    ndarray<T> flat = this->flatten();
    long long idx = 0;
    T minVal = flat[0];
    for(long long i;i<this->_size;++i){
        if(flat[i] < minVal){
            minVal = flat[i];
            idx = i;
        }
    }

    return idx;
}

template <typename T>
ndarray<int> ndarray<T>::argmax(int axis){
    return this->argreduction(axis, argmax_reduction);
}

template <typename T>
ndarray<int> ndarray<T>::argmin(int axis){
    return this->argreduction(axis, argmin_reduction);
}

// reduction method for argmax(). argmin()
template <typename T>
ndarray<int> ndarray<T>::argreduction(int axis,bool (*func)(T &a, T&b)){

    vector<int> __shape = __reduce_shape(this->_shape, this->_ndim, axis);
    long long __size = __reduce_size(this->_shape, this->_ndim, axis);

    // initialization
    vector<int> arr = vector<int>(__size,0);
    ndarray<int> trans(arr,__shape);
    // adjust elements
    // get a copy of ndarray
    ndarray<T> copy(this->_data,this->_shape,this->_strides,this->_axes);
    // add a dimension
    vector<int> n_shape(this->_shape);
    n_shape.emplace_back(1);
    copy = copy.reshape(n_shape);

    // transpose axis
    vector<int> n_axes;
    for(int i=0;i<this->_ndim+1;++i) n_axes.emplace_back(i);
    swap(n_axes[_ndim],n_axes[axis]);
    copy = copy.transpose(n_axes);

    // delete the additional dimension
    copy = copy.squeeze();
    // flatten the array
    copy = copy.flatten();

    // assign elements
    int step = this->_shape[axis];
    for(long long i=0;i<__size;++i){
        T Val = copy[i*step];
        int idx = 0;
        for(int j=0;j<step;++j){
            if(func(copy[i*step + j],Val)){
                Val = copy[i*step + j];
                idx = j;
            }
        }
        trans[i] = idx;
    }

    return trans;
}

