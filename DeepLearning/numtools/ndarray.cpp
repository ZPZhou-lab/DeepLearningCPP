#include <bits/stdc++.h>
#include <bits/types/clock_t.h>
#include <cassert>
#include <cfloat>
#include <climits>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <numeric>
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

template <typename _Tp>
class ndarray{
private:
    // define _data type
    typedef _Tp value_type; 
    value_type _dtype;

    // store all elements in a vector
    vector<_Tp> _data; 
    // the number of elements
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
    // inplace operations update
    void __inplace_change(vector<_Tp> &array, vector<int> &shape, vector<int> &strides, vector<int> &axes);

    // internal(private) access method, check is omitted
    _Tp __item(long long args);
    _Tp __item(vector<int>& args);
    vector<int> __item_loc(long long args, vector<int> axis=vector<int>());

    // reuction method for sum(), max(), min()
    template<typename T1>
    ndarray<T1> _reduction(vector<int> axis, void (*func)(T1 &a,_Tp &b), bool keepdim, vector<T1> &arr, 
                       long long step, vector<int> &__shape, long long __size);
    // reduction method for argmax(), argmin()
    ndarray<int> _argreduction(int axis, bool (*func)(_Tp &a, _Tp &b));

    // function for broadcast
    template <typename T1>
    pair<ndarray<_Tp>,ndarray<T1>> _broadcast_flatten(ndarray<T1> &b);

public:
    // default constructer
    ndarray();
    // constructer
    ndarray(vector<_Tp>& array, vector<int>& shape);
    ndarray(vector<_Tp>& array, vector<int>& shape, vector<int>& strides, vector<int>& axes);
    // destructor
    ~ndarray();

    // get a copy
    ndarray copy(void);
    // get a copy and change data type
    template<typename _Tp2>
    ndarray<_Tp2> astype(void);
    
    // access element
    _Tp &item(long long args); // access element by flat index
    _Tp &item(vector<int>& args); // access element by an nd-index into the array
    
    _Tp at(long long idx); // access element by index

    // return _data type
    const string dtype(void) const;
    // return the number of elements
    const long long size(void) const;
    // return the number of dimensions
    const int ndim(void) const;
    // return the shape of the nd-array
    const vector<int> shape(void) const;
    // return the strides of the nd-array
    const vector<int> strides(void) const;
    // fetch _data
    vector<_Tp> &data(void);

    // array transform
    ndarray transpose(vector<int>& axes, bool inplace=false);
    ndarray reshape(vector<int>& shape, bool inplace=false);
    ndarray flatten(bool inplace=false);
    ndarray squeeze(vector<int> axis=vector<int>(), bool inplace=false);
    ndarray expand_dims(vector<int> axis, bool inplace=false);
    ndarray expand_dims(int axis, bool inplace=false);
    ndarray T(bool inplace=false);
    ndarray repeat(int repeats, int axis);
    ndarray repeat(int repeats);

    // array operation
    ndarray sum(vector<int> axis, bool keepdim=false);
    _Tp sum(void);
    ndarray max(vector<int> axis, bool keepdim=false);
    _Tp max(void);
    ndarray min(vector<int> axis, bool keepdim=false);
    _Tp min(void);
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
    
    // sort operation
    void sort(void);
    void sort(int axis);
    ndarray<long long> argsort(void);
    ndarray<long long> argsort(int axis);

    // matrix production, i.e. dot method()
    template<typename T1>
    ndarray<double> dot(ndarray<T1> &mat);

    // clip the value
    ndarray clip(_Tp min, _Tp max, bool inplace=false);

    // show matrix
    void show(void);

    // opetator reload
    // operator reload []
    _Tp &operator [] (long long args);
    const _Tp &operator [] (long long args) const;
    template<typename ...Args>
    _Tp &operator () (Args...args);
    template<typename ...Args>
    const _Tp &operator () (Args...args) const;

    // operation between ndarray and real number
    ndarray operator - (const double b);
    ndarray<double> operator * (const double b);
    ndarray<double> operator / (const double b);
    ndarray operator + (const double b);
    template<typename _Tp2>
    friend ndarray<double> operator+(const double a, const ndarray<_Tp2> &b);
    template<typename _Tp2>
    friend ndarray<double> operator-(const double a, const ndarray<_Tp2> &b);
    template<typename _Tp2>
    friend ndarray<double> operator*(const double a, const ndarray<_Tp2> &b);
    template<typename _Tp2>
    friend ndarray<double> operator/(const double a, const ndarray<_Tp2> &b);

    // operation between ndarray and ndarray
    template<typename T1>
    ndarray operator + (ndarray<T1> &b);
    template<typename T1>
    ndarray operator - (ndarray<T1> &b);
    template<typename T1>
    ndarray<double> operator * (ndarray<T1> &b);
    template<typename T1>
    ndarray<double> operator / (ndarray<T1> &b);

    // subarray method
    ndarray subarr(const vector<int>& indices);

};

// default constructer
template <typename _Tp>
ndarray<_Tp>::ndarray(){
    this->_ndim = 0;
    this->_size = 0;
}

// constructer
template <typename _Tp>
ndarray<_Tp>::ndarray(vector<_Tp>& array, vector<int>& shape){
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
template <typename _Tp>
ndarray<_Tp>::ndarray(vector<_Tp>& array, vector<int>& shape, vector<int>& strides, vector<int>& axes){
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
template <typename _Tp>
ndarray<_Tp>::~ndarray(){
    cout<<"Destructor called"<<endl;
}

// operator reload
template <typename _Tp>
ndarray<double> ndarray<_Tp>::operator/(const double b){
    vector<double> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] /= b;

    ndarray<double> trans(res,this->_shape);

    return trans;
}

template<typename _Tp>
ndarray<_Tp> ndarray<_Tp>::operator+(const double b){
    vector<_Tp> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] += b;

    ndarray<_Tp> trans(res,this->_shape);

    return trans;
}

template<typename _Tp2>
ndarray<double> operator+(const double a, ndarray<_Tp2> &b){
    return b + a;
}
template<typename _Tp2>
ndarray<double> operator-(const double a, ndarray<_Tp2> &b){
    return b - a;
}
template<typename _Tp2>
ndarray<double> operator*(const double a, ndarray<_Tp2> &b){
    return b * a;
}
template<typename _Tp2>
ndarray<double> operator/(const double a, ndarray<_Tp2> &b){
    return b / a;
}

template <typename _Tp>
ndarray<double> ndarray<_Tp>::operator*(const double b){
    vector<double> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] *= b;

    ndarray<double> trans(res,this->_shape);

    return trans;
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::operator-(const double b){
    vector<_Tp> res(this->_data);
    for(int i=0;i<this->_size;++i) res[i] -= b;

    ndarray<_Tp> trans(res,this->_shape);

    return trans;
}

template <typename _Tp>
template <typename T1>
ndarray<_Tp> ndarray<_Tp>::operator+(ndarray<T1> &b){
    // check operator shape
    bool broadcast = __check_operator_shape(this->shape(),b.shape());
    // init
    ndarray<_Tp> trans;

    if(broadcast){
        auto flat = _broadcast_flatten(b);
        return flat.first + flat.second;
    }else{
        ndarray<_Tp> flat1 = this->flatten();
        ndarray<T1> flat2 = b.flatten();
        vector<_Tp> array(flat1.data());
        for(int i=0;i<this->_size;++i) array[i] += flat2[i];

        trans = ndarray<_Tp>(array,this->_shape);
    }

    return trans;
}

template <typename _Tp>
template <typename T1>
ndarray<_Tp> ndarray<_Tp>::operator-(ndarray<T1> &b){
    // check operator shape
    bool broadcast = __check_operator_shape(this->shape(),b.shape());
    // init
    ndarray<_Tp> trans;

    if(broadcast){
        auto flat = _broadcast_flatten(b);
        return flat.first - flat.second;
    }else{
        ndarray<_Tp> flat1 = this->flatten();
        ndarray<T1> flat2 = b.flatten();
        vector<_Tp> array(flat1.data());
        for(int i=0;i<this->_size;++i) array[i] -= flat2[i];

        trans = ndarray<_Tp>(array,this->_shape);
    }

    return trans;
}

template <typename _Tp>
template <typename T1>
ndarray<double> ndarray<_Tp>::operator*(ndarray<T1> &b){
    // check operator shape
    bool broadcast = __check_operator_shape(this->shape(),b.shape());
    // init
    ndarray<double> trans;

    if(broadcast){
        auto flat = _broadcast_flatten(b);
        return flat.first * flat.second;
    }else{
        ndarray<_Tp> flat1 = this->flatten();
        ndarray<T1> flat2 = b.flatten();
        vector<double> array(flat1.data());
        for(int i=0;i<this->_size;++i) array[i] *= flat2[i];

        trans = ndarray<double>(array,this->_shape);
    }

    return trans;
}

template <typename _Tp>
template <typename T1>
ndarray<double> ndarray<_Tp>::operator/(ndarray<T1> &b){
    // check operator shape
    bool broadcast = __check_operator_shape(this->shape(),b.shape());
    // init
    ndarray<double> trans;

    if(broadcast){
        auto flat = _broadcast_flatten(b);
        return flat.first / flat.second;
    }else{
        ndarray<_Tp> flat1 = this->flatten();
        ndarray<T1> flat2 = b.flatten();
        vector<double> array(flat1.data());
        for(int i=0;i<this->_size;++i) array[i] /= flat2[i];

        trans = ndarray<double>(array,this->_shape);
    }

    return trans;
}


// maintenance function of shape_cumprod
template <typename _Tp>
void ndarray<_Tp>::__update_shape_cumprod(void){
    this->__shape_cumprod = vector<long long>(_ndim,1);
    for(int i=_ndim-1;i>0;i--) this->__shape_cumprod[i-1] = this->__shape_cumprod[i] * this->_shape[i];
}

// array update when inplace operations
template <typename _Tp>
void ndarray<_Tp>::__inplace_change(vector<_Tp> &array, vector<int> &shape, vector<int> &strides, vector<int> &axes){
    // update array
    this->_data = array;
    // update array information
    this->_shape = shape;
    this->_ndim = shape.size();
    this->_ndim = shape.size();

    this->_strides = strides;
    this->_axes = axes;

    __update_shape_cumprod();
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
template <typename _Tp>
inline vector<int> ndarray<_Tp>::__item_loc(long long args, vector<int> axis){
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
template <typename _Tp>
inline _Tp &ndarray<_Tp>::item(long long args){
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
template <typename _Tp>
inline _Tp &ndarray<_Tp>::item(vector<int>& args){
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
template <typename _Tp>
inline _Tp ndarray<_Tp>::at(long long idx){
    // return element
    return this->_data[idx];
}

template<typename _Tp>
inline _Tp &ndarray<_Tp>::operator[](long long args){
    return this->_data[args];
}

template <typename _Tp>
const _Tp &ndarray<_Tp>::operator[](long long args) const{
    return this->_data[args];
}

template<typename _Tp>
template<typename ...Args>
inline _Tp &ndarray<_Tp>::operator()(Args...args){
    vector<int> idx;
    idx = fetchArgs(idx,args...);
    return this->item(idx);
}

template<typename _Tp>
template<typename ...Args>
const _Tp &ndarray<_Tp>::operator()(Args...args) const{
    vector<int> idx;
    idx = fetchArgs(idx,args...);
    return this->item(idx);
}

// return _data type
template <typename _Tp>
const string ndarray<_Tp>::dtype(void) const{
    map<string,string> dtypes = {
        {"i","int"}, {"f","float"}, {"d","double"}, {"l","long"}, {"b","bool"},
        {"e", "long double"}, {"x","long long"}
    };
    const type_info &_dataInfo = typeid(this->_dtype);
    return dtypes[_dataInfo.name()];
}

// the number of elements
template <typename _Tp>
const long long ndarray<_Tp>::size(void) const{
    return this->_size;
}

// the number of dimensions
template <typename _Tp>
const int ndarray<_Tp>::ndim(void) const{
    return this->_ndim;
}

// the number of elements in each dimension
template <typename _Tp>
const vector<int> ndarray<_Tp>::shape(void) const{
    return this->_shape;
}

// the strides of the nd-array
template <typename _Tp>
const vector<int> ndarray<_Tp>::strides(void) const{
    return this->_strides;
}

// fetch data
template <typename _Tp>
vector<_Tp> &ndarray<_Tp>::data(void){
    return this->_data;
}

// print the array
template <typename _Tp>
void ndarray<_Tp>::show(void){
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
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::transpose(vector<int>& axes, bool inplace){
    // anomaly detection
    __check_axes(axes, this->_ndim);

    // update strides
    vector<int> __strides = permute(this->_strides, axes);
    // update shape
    vector<int> __shape = permute(this->_shape, axes);
    // update axes
    vector<int> __axes = permute(this->_axes, axes);
    
    // transformed array
    ndarray<_Tp> trans(this->_data,__shape,__strides,__axes);
    // if inplace, then change the array itself
    if(inplace)  __inplace_change(this->_data, __shape, __strides, __axes);

    return trans;
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::T(bool inplace){
    // set axes
    vector<int> axes;
    for(int i=this->_ndim-1;i>=0;--i) axes.emplace_back(i);

    return this->transpose(axes,inplace);
}

// reshape
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::reshape(vector<int> &shape, bool inplace){
    // anomaly detection
    long long check_size = 1;
    for(auto s:shape) check_size *= s;

    // judge whether the number of array elements matches the dimension
    __check_shape(this->_size,check_size);

    // initialization
    ndarray<_Tp> trans;

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
        vector<_Tp> array(this->_data);

        // element table switching
        for(long long i=0;i<this->_size;++i){
            array[i] = item(i);
        }

        trans = ndarray<_Tp>(array,shape,__strides,__axes);
        // if inplace, then change the array itself
        if(inplace)  __inplace_change(array, shape, __strides, __axes);
    }
    // do not need to adjust the position of elements
    else{
        trans = ndarray<_Tp>(this->_data,shape,__strides,__axes);
        // if inplace, then change the array itself
        if(inplace)  __inplace_change(this->_data, shape, __strides, __axes);
    }

    return trans;
}

// flatten
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::flatten(bool inplace){
    vector<int> shape = {(int)this->_size};
    return reshape(shape,inplace);
}

// squeeze
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::squeeze(vector<int> axis, bool inplace){
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

    ndarray<_Tp> trans(this->_data,__shape,__strides,__axes_idx);
    // if inplace, then change the array itself
    if(inplace) __inplace_change(this->_data, __shape, __strides, __axes_idx);

    return trans;
}

// expand dims
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::expand_dims(vector<int> axis, bool inplace){
    /*
    Expand the shape of an array.
    Insert a new axis that will appear at the `axis` position in the expanded array shape.
    axis represents the position in the expanded axes where the new axis (or axes) is placed.
    */
    
    // check expand dims
    __check_expand(this->_ndim,axis);
    // new ndim
    int n_ndim = this->_ndim + axis.size();
    // init new shape
    vector<int> n_shape(n_ndim,0);
    int ptr1 = 0, ptr2 = 0;
    for(int i=0;i<n_ndim;++i){
        if(i == axis[ptr2]){
            n_shape[i] = 1;
            ptr2++;
        }else{
            n_shape[i] = this->_shape[ptr1];
            ptr1++;
        }
    }
    // do reshape
    return this->reshape(n_shape,inplace);
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::expand_dims(int axis, bool inplace){
    /*
    Expand the shape of an array.
    Insert a new axis that will appear at the `axis` position in the expanded array shape.
    axis represents the position in the expanded axes where the new axis (or axes) is placed.
    */
    
    // use method expand_dims(vector<int> axis, bool inplace)
    // init axis
    vector<int> __axis = {axis};
    // do reshape
    return this->expand_dims(__axis,inplace);
}

// sum
template <typename _Tp>
_Tp ndarray<_Tp>::sum(void){
    _Tp s = 0;
    for(auto e:this->_data) s += e;

    return s;
}

// max
template <typename _Tp>
_Tp ndarray<_Tp>::max(void){
    _Tp s = INT_MIN;
    for(auto e:this->_data) s = std::max(s,e);

    return s;
}

// min
template <typename _Tp>
_Tp ndarray<_Tp>::min(void){
    _Tp s = INT_MAX;
    for(auto e:this->_data) s = std::min(s,e);

    return s;
}

// any()
template <typename _Tp>
bool ndarray<_Tp>::any(void){
    bool flag = false;
    for(auto e:this->_data) flag = flag || e;

    return flag;
}

// all()
template <typename _Tp>
bool ndarray<_Tp>::all(void){
    bool flag = true;
    for(auto e:this->_data) flag = flag && e;

    return flag;
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::sum(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<_Tp> arr(__size,0);
    return this->_reduction(axis, sum_reduction, keepdim, arr, step, __shape, __size);
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::max(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<_Tp> arr(__size,INT_MIN);
    return this->_reduction(axis, max_reduction, keepdim, arr, step, __shape, __size);
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::min(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<_Tp> arr(__size,INT_MAX);
    return this->_reduction(axis, min_reduction, keepdim, arr, step, __shape, __size);
}

template <typename _Tp>
ndarray<int> ndarray<_Tp>::any(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<int> arr(__size,0);
    return this->_reduction(axis, any_reduction, keepdim, arr, step, __shape, __size);
}

template <typename _Tp>
ndarray<int> ndarray<_Tp>::all(vector<int> axis, bool keepdim){
    // update nwe shape, size and step
    long long __size = __reduce_size(this->_shape,this->_ndim,axis);
    vector<int> __shape = __reduce_shape(this->_shape,this->_ndim,axis);
    long long step = __reduce_step(this->_shape,this->_ndim,axis);
    // init array
    vector<int> arr(__size,1);
    return this->_reduction(axis, all_reduction, keepdim, arr, step, __shape, __size);
}

// reduction method for sum(), min(), max()
template <typename _Tp>
template <typename T1>
ndarray<T1> ndarray<_Tp>::_reduction(vector<int> axis, void (*func)(T1 &a, _Tp &b), bool keepdim, vector<T1> &arr,
                                   long long step, vector<int> &__shape,long long __size){
    // initialization
    ndarray<T1> trans(arr,__shape);
    // adjust elements
    // get a copy of ndarray
    ndarray<_Tp> copy(this->_data,this->_shape,this->_strides,this->_axes);
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
template <typename _Tp>
double ndarray<_Tp>::mean(void){
    double m;

    _Tp s = this->sum();
    m = s / this->_size;

    return m;
}

// mean
template <typename _Tp>
ndarray<double> ndarray<_Tp>::mean(vector<int> axis, bool keepdim){
    // figure sum
    ndarray<_Tp> s = this->sum(axis,keepdim);

    // cimpute number of elements
    long long __num = 1;
    for(auto i:axis) __num *= this->_shape[i];    

    // compute mean
    ndarray<double> trans = s / __num;

    return trans;
}

// argmax
template <typename _Tp>
long long ndarray<_Tp>::argmax(void){
    ndarray<_Tp> flat = this->flatten();
    long long idx = 0;
    _Tp maxVal = flat[0];
    for(long long i=0;i<this->_size;++i){
        if(flat[i] > maxVal){
            maxVal = flat[i];
            idx = i;
        }
    }

    return idx;
}

// argmin
template <typename _Tp>
long long ndarray<_Tp>::argmin(void){
    ndarray<_Tp> flat = this->flatten();
    long long idx = 0;
    _Tp minVal = flat[0];
    for(long long i;i<this->_size;++i){
        if(flat[i] < minVal){
            minVal = flat[i];
            idx = i;
        }
    }

    return idx;
}

template <typename _Tp>
ndarray<int> ndarray<_Tp>::argmax(int axis){
    return this->_argreduction(axis, argmax_reduction);
}

template <typename _Tp>
ndarray<int> ndarray<_Tp>::argmin(int axis){
    return this->_argreduction(axis, argmin_reduction);
}

// reduction method for argmax(). argmin()
template <typename _Tp>
ndarray<int> ndarray<_Tp>::_argreduction(int axis, bool (*func)(_Tp &a, _Tp&b)){

    vector<int> __shape = __reduce_shape(this->_shape, this->_ndim, axis);
    long long __size = __reduce_size(this->_shape, this->_ndim, axis);

    // initialization
    vector<int> arr = vector<int>(__size,0);
    ndarray<int> trans(arr,__shape);
    // adjust elements
    // get a copy of ndarray
    ndarray<_Tp> copy(this->_data,this->_shape,this->_strides,this->_axes);
    // add a dimension
    copy = copy.expand_dims(this->_ndim);

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
        _Tp Val = copy[i*step];
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

// sort method
template <typename _Tp>
void ndarray<_Tp>::sort(void){
    /*
    The default sort(void) method, the array is flattened before.
    */
    // flat the array
    this->flatten(true);
    // do sort
    quicksort(this->_data,0,this->_size-1);
}

template <typename _Tp>
void ndarray<_Tp>::sort(int axis){
    /*
    Axis along which to sort. if axis is -1, sort() will sorts array along the last axis.
    */
    // check axis
    if(axis == -1) axis = this->_ndim - 1;
    __check_axis(this->_ndim,axis);

    long long step = this->_shape[axis];

    // add a dimension
    vector<int> n_shape(this->_shape);
    n_shape.emplace_back(1);
    this->reshape(n_shape,true);

    // transpose axis
    vector<int> n_axes;
    for(int i=0;i<this->_ndim;++i) n_axes.emplace_back(i);
    swap(n_axes[_ndim-1],n_axes[axis]);
    this->transpose(n_axes,true);

    // delete the additional dimension
    vector<int> n_axis;
    n_shape = this->_shape;
    this->squeeze(n_axis,true);

    // flatten the array
    this->flatten(true);

    // sort elements
    for(long long i=0;i<(long long)(this->_size / step);++i){
        quicksort(this->_data, i*step, (i+1)*step-1);
    }

    // recover the shape
    // reverse the squeeze
    this->reshape(n_shape,true);
    // reverse the transpose
    this->transpose(n_axes,true);
    // reverse add a dimension
    this->squeeze(n_axis,true);
}

// argsort method
template <typename _Tp>
ndarray<long long> ndarray<_Tp>::argsort(void){
    /*
    The default argsort(void) method, the array is flattened before.
    */
    // flat the array
    auto flat = this->flatten();
    
    // construct sort index
    vector<long long> idx(flat.size());
    for(long long i=0;i<flat.size();++i) idx[i] = i;
    vector<int> __shape = {flat.size()};

    // do sort
    quicksort(flat, idx, 0, flat.size()-1);

    ndarray<long long> sorted_idx(idx,__shape);
    return sorted_idx;
}

template <typename _Tp>
ndarray<long long> ndarray<_Tp>::argsort(int axis){
    /*
    Axis along which to argsort. if axis is -1, argsort() will sorts array along the last axis.
    */
    // check axis
    if(axis == -1) axis = this->_ndim - 1;
    __check_axis(this->_ndim,axis);

    long long step = this->_shape[axis];

    // copy array and add a dimension
    vector<int> n_shape(this->_shape);
    n_shape.emplace_back(1);
    auto copy = this->reshape(n_shape);

    // transpose axis
    vector<int> n_axes;
    for(int i=0;i<this->_ndim+1;++i) n_axes.emplace_back(i);
    swap(n_axes[_ndim],n_axes[axis]);
    copy = copy.transpose(n_axes);

    // delete the additional dimension
    vector<int> n_axis;
    n_shape = copy.shape();
    copy = copy.squeeze();

    // flatten the array
    copy = copy.flatten();

    // init index
    vector<long long> idx(this->_size,0);
    // sort elements
    for(long long i=0;i<(long long)(this->_size / step);++i){
        // set init index
        for(long long j=0;j<step;++j) idx[i*step+j] = j;
        // do sort
        quicksort(copy, idx, i*step, (i+1)*step - 1);
    }

    // construct array
    // reverse the squeeze
    ndarray<long long> trans(idx,n_shape);
    // reverse the transpose
    trans = trans.transpose(n_axes);
    // reverse add a dimension
    trans = trans.squeeze();

    return trans;
}

// cilp
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::clip(_Tp min, _Tp max, bool inplace){
    // copy array
    vector<_Tp> array(this->_data);

    // clip the elements
    for(auto &a:array){
        if(a < min){
            a = min;
        }else if(a > max){
            a = max;
        }
    }

    // create ndarray
    ndarray<_Tp> trans(array,this->_shape,this->_strides,this->_axes);
    if(inplace) this->_data = array;

    return trans;
}

// repeat
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::repeat(int repeats){
    // check repeats
    __check_repeat(repeats);

    // flat the array
    auto flat = this->flatten();
    vector<_Tp> array(flat.size() * repeats,0);

    for(long long i=0;i<flat.size();++i){
        for(int j=0;j<repeats;++j){
            array[i*repeats+j] = flat[i];
        }
    }

    // create array
    vector<int> __shape = {flat.size()*repeats};
    ndarray<_Tp> trans(array,__shape);
    
    return trans;
}

template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::repeat(int repeats, int axis){
    // check repeats
    __check_repeat(repeats);

    // init the array
    long long __size = this->_size * repeats;
    vector<_Tp> array(__size,0);
    // flat the array
    auto flat = this->flatten();
    flat = flat.reshape(this->_shape);
    // fetch strides
    vector<int> __strides(flat.strides());
    
    // assign elements
    for(long long i=0;i<this->_size;++i){
        long long idx = (i / __strides[axis]) * repeats * __strides[axis] + (i % __strides[axis]); 
        for(int j=0;j<repeats;++j){
            array[idx + __strides[axis]*j] = flat[i];
        }
    }

    // update new shape
    vector<int> __shape(this->_shape);
    __shape[axis] *= repeats;

    ndarray<_Tp> trans(array,__shape);
    
    return trans;
}


// matrix production, dot method()
template <typename _Tp>
template <typename T1>
ndarray<double> ndarray<_Tp>::dot(ndarray<T1> &mat){
    /*
    dot(a, b): Dot product of two arrays. Specifically,

    case 1
    - If both `a` and `b` are 1-D arrays, it is inner product of vectors (without complex conjugation).
    case 2
    - If both `a` and `b` are 2-D arrays, it is matrix multiplication, but using `matmul` is preferred.
    case 3
    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over the last axis of `a` and `b`.
    case 4
    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a sum product over the last axis of `a` and the second-to-last axis of `b`::

    */ 

    // init
    ndarray<double> trans;

    // case 1
    if(this->_ndim == 1 && mat.ndim() == 1){
        // check size
        __check_dot(this->_size,mat.size());

        // init array
        vector<double> res(1,0);
        // compute inner product
        res[0] = __inner_product(this->_data,0,mat.data(),0,this->_size);
        vector<int> __shape = {1};

        trans = ndarray<double>(res,__shape);
    }
    // case 3
    else if(this->_ndim > 1 && mat.ndim() == 1){
        // check size
        __check_dot(this->_shape[this->_ndim-1],mat.size());

        // compute new shape and new size
        long long step = mat.size();
        long long __size = this->_size / step;
        
        // delete the last dim
        vector<int> __shape(this->_shape);
        __shape.pop_back();

        // init array
        vector<double> res(__size);
        // flatten the array
        auto flat = this->flatten();
        
        // compute inner product
        for(long long i=0;i<__size;++i) res[i] = __inner_product(flat.data(),i*step,mat.data(),0,step);
        
        // create ndarray
        trans = ndarray<double>(res,__shape);

    }
    // case 2
    else if(this->_ndim == 2 && mat.ndim() == 2){
        // check size
        __check_dot(this->_shape[1], mat.shape()[0]);

        // compute matrix production
        vector<double> res = __matrix_prod(this->flatten().data(),mat.T().flatten().data(),mat.shape()[0]);
        // init new shape
        vector<int> __shape = {this->_shape[0],mat.shape()[1]};

        // create ndarray
        trans = ndarray<double>(res,__shape);
    }
    // case 4
    else{
        /*
        对于case 4，应该将其先转换为两个二维矩阵相乘，然后借用快速矩阵乘法，最后再reshape返回到正确的形状
        将元素放置到二维矩阵所需要的时间复杂度与元素个数相同
        */
        // check size
        __check_dot(this->_shape[this->_ndim-1], mat.shape()[mat.ndim()-2]);

        // compute new shape and size
        vector<int> __shape;
        long long __size = 1;
        for(int i=0;i<this->_ndim-1;++i){
            __shape.emplace_back(this->_shape[i]);
            __size *= this->_shape[i];
        }

        // transpose the axes of mat
        vector<int> axes;
        for(int i=0;i<mat.ndim();++i) axes.emplace_back(i);
        std::swap(axes[mat.ndim()-1],axes[mat.ndim()-2]);
        auto mat2 = mat.transpose(axes);

        for(int i=0;i<mat2.ndim()-1;++i){
            __shape.emplace_back(mat2.shape()[i]);
            __size *= mat2.shape()[i];
        }

        // init array
        vector<double> res(__size,0);
        
        // flatten
        auto mat1 = this->flatten();
        mat2 = mat2.flatten();
        // step for inner product
        long long step = this->_shape[this->_ndim-1];

        long long size1 = mat1.size() / step, size2 = mat2.size() / step;

        // create two 2-D array
        vector<int> __shape1 = {(int)size1,(int)step};
        mat1 = ndarray<_Tp>(mat1.data(),__shape1);
        vector<int> __shape2 = {(int)step,(int)size2};
        mat2 = ndarray<T1>(mat2.data(),__shape2);

        // compute matrix product, use case 2 for fast matrix multiply
        // do reshape
        trans = mat1.dot(mat2).reshape(__shape);
        
    }


    return trans;
}


// function for broadcast
template <typename _Tp>
template <typename T1>
pair<ndarray<_Tp>, ndarray<T1>> ndarray<_Tp>::_broadcast_flatten(ndarray<T1> &b){
    int s1 = this->shape().size(), s2 = b.shape().size();
    int s = std::min(s1,s2);
    vector<int> _shape1 = this->shape(), _shape2 = b.shape();
    // init
    ndarray<_Tp> flat1 = this->copy();
    ndarray<T1> flat2 = b.copy();

    // do broadcast for trailing dimension
    for(int i=0;i<s;++i){
        // broadcast when dimensions are not equal
        if(_shape1[s1 - 1 - i] != _shape2[s2 - 1 - i]){
            // broadcast the array with dimension 1
            if(_shape1[s1 - 1 - i] == 1){
                flat1 = flat1.repeat(_shape2[s2 - 1 - i],s1 - 1 - i);
            }else{
                flat2 = flat2.repeat(_shape1[s1 - 1 - i],s2 - 1 - i);
            }
        }
    }

    // do broadcast for head dimension
    if(s1 > s2){
        for(int i=0;i<s1 - s2;++i){
            flat2.expand_dims(0,true);
            flat2 = flat2.repeat(_shape1[s1 - s2 - 1 - i],0);
        }
    }else if(s1 < s2){
        for(int i=0;i<s2 - s1;++i){
            flat1.expand_dims(0,true);
            flat1 = flat1.repeat(_shape2[s2 - s1 - 1 - i],0);
        }
    }

    return make_pair(flat1,flat2);
}

// get a copy
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::copy(void){
    vector<_Tp> data_copy(this->_data);
    return ndarray<_Tp>(data_copy,this->_shape,this->_strides,this->_axes);
}

// get a copy and change data type
template <typename _Tp>
template <typename _Tp2>
ndarray<_Tp2> ndarray<_Tp>::astype(void){
    // init data vector
    vector<_Tp2> n_data(this->_size);
    // change value type
    for(long long i=0;i<this->_size;++i) n_data[i] = (_Tp2)this->_data[i];
    
    // create ndarray
    ndarray<_Tp2> trans(n_data,this->_shape,this->_strides,this->_axes);

    return trans;
}

// subarray method
template <typename _Tp>
ndarray<_Tp> ndarray<_Tp>::subarr(const vector<int>& indices){
    __check_subarr(this->_shape,indices);
    // init
    auto subarray = ndarray<_Tp>();
    
    if(indices.size() == 2){
        long long start = indices[0], end = indices[1];
        vector<_Tp> data = vector<_Tp>(end - start,0);
        // assign elements
        for(long long i=start;i<end;++i) data[i] = this->_data[i];
        vector<int> __shape = {(int)(end - start),1};
        subarray = ndarray<_Tp>(data,__shape);
    }
    if(indices.size() == 4){
        int row_s = indices[0], row_e = indices[1], col_s = indices[2], col_e = indices[3];
        // init
        vector<_Tp> data = vector<_Tp>((row_e - row_s)*(col_e - col_s),0);
        vector<int> __shape = {(row_e - row_s), (col_e - col_s)};
        subarray = ndarray<_Tp>(data,__shape);
        // assign elements
        vector<int> idx;
        for(int i=row_s;i<row_e;++i){
            for(int j=col_s;j<col_e;++j){
                idx = {i,j};
                subarray(i - row_s,j - col_s) = this->item(idx);
            }
        }
    }

    return subarray;
}

// The interface of each method should be adjusted
// and the `const` or `reference modifier &` of the response should be added