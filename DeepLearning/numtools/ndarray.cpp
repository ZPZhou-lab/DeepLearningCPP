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

    void change(long long iloc, T num); // 改变某个位置处元素的值

    vector<long long> idx_prod(void);
    // 维度变换
    ndarray transpose(vector<int>& axes);
    ndarray reshape(vector<int>& shape);
    ndarray flatten(void);
    
    // 打印矩阵
    void show(void);
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

// 坐标置换辅助函数
vector<long> permutation(vector<long>& at, vector<int>& axes){
    vector<long> permuted(at.size());
    for(int i=0;i<(int)at.size();++i) permuted[i] = at[axes[i]];
    return permuted;
}

// 生成所有可能的排列
void permutation_DFS(vector<vector<long>>& orders, vector<long>& path, vector<long>& shape, int axis){
    if(axis == (int)shape.size()){
        orders.emplace_back(path);
        return;
    }
    for(int i=0;i<shape[axis];++i){
        // 添加元素
        path.emplace_back(i);
        // 递归搜索
        permutation_DFS(orders, path, shape, axis+1);
        // 恢复状态
        path.pop_back();
    }
}
vector<vector<long>> permutation_generator(vector<long>& shape){
    vector<vector<long>> orders;
    vector<long> path;
    permutation_DFS(orders,path,shape,0);
    return orders;
}

// 根据索引判断位置
template <typename ...Args>
long long to_iloc(vector<long long> idx_prod, int ndim, long long size,Args...args){
    // 取出索引
    vector<int> idx;
    idx = fetchArgs(idx,args...);
    // 判断维度是否正确
    assert((int)idx.size() == ndim);
    long long loc = 0;
    for(int i=0;i<ndim;++i) loc += idx_prod[ndim-i-1]*idx[i];
    assert(loc >= 0 && loc < size);
    return loc;
}
// 另一种访问元素位置的模板
long long to_iloc(vector<long long> idx_prod, int ndim, long long size, vector<long> at){
    long long loc = 0;
    for(int i=0;i<ndim;++i) loc += idx_prod[ndim-i-1]*at[i];
    assert(loc >= 0 && loc < size);
    return loc;
}

// 把位置转换为索引
vector<long> to_at(long long iloc, vector<long long> idx_prod, int ndim){
    vector<long> at;
    for(int i=0;i<ndim;++i){
        long s = iloc / idx_prod[ndim-1-i];
        at.emplace_back(s);
        iloc %= idx_prod[ndim-1-i];
    }
    return at;
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
    long long loc = to_iloc(this->_idx_prod,this->_ndim,this->_size,args...);
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

// 返回各维度累计乘积
template <typename T>
vector<long long> ndarray<T>::idx_prod(void){
    return this->_idx_prod;
}

// 改变某个位置处元素的值
template <typename T>
void ndarray<T>::change(long long iloc, T num){
    this->data[iloc] = num;
}

template <typename T>
void ndarray<T>::show(void){
    if(this->_ndim == 1){
        for(auto d : data) printf("%+12.4f", d);
        printf("\n");
    }else{
        for(long long i=0;i<this->_size;++i){
            printf("%+12.4f", this->data[i]);
            for(int j=1;j<this->_ndim;++j){
                if((i+1)%this->_idx_prod[j] == 0) printf("\n");
            }
        }
    }

}

// 维度变换
template <typename T>
ndarray<T> ndarray<T>::transpose(vector<int>& axes){
    // 异常检测
    try{
        if((int)axes.size() != this->_ndim){
            throw "axes don't match array";
        }
        set<int> tmp;
        for(auto axis : axes){
            if(axis < -this->_ndim || axis >= this->_ndim){
                throw "axis is out of bounds for array of dimension ";
            }
            if(axis < 0){
                axis = this->_ndim - axis;
            }
            if(tmp.find(axis) != tmp.end()){
                throw "repeated axis in transpose";
            }
            tmp.insert(axis);
        }
    }catch(const char* msg){
        cout<<msg<<endl;
        assert(false);
    }
    // 开始计算矩阵转置
    for(int i=0;i<(int)axes.size();++i) axes[i] = axes[i] < 0 ? this->_ndim-axes[i] : axes[i];
    // 构造新的数据，赋予新的维度
    vector<long> new_shape = permutation(this->_shape,axes);
    ndarray<T> trans(this->data, new_shape);
    // 生成所有可能的排列
    set<long long> permuted;
    for(long long i=0;i<this->_size;++i){
        long long from = i;
        vector<long> at = to_at(i,this->_idx_prod,this->_ndim);
        long long to = to_iloc(trans.idx_prod(),trans.ndim(),trans.size(),permutation(at,axes));
        if(from == to){
            permuted.insert(to);
            continue;
        }
        while(permuted.find(to) == permuted.end()){
            trans.change(to, this->data[from]);
            permuted.insert(to);
            from = to;
            at = to_at(from,this->_idx_prod,this->_ndim);
            to = to_iloc(trans.idx_prod(),trans.ndim(),trans.size(),permutation(at,axes));
        }
    }
    // 替换矩阵维度
    this->_shape = permutation(this->_shape,axes);
    return trans;
}