#include <bits/stdc++.h>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <vector>
#include <typeinfo>
#include <stdarg.h>
using namespace std;


// check when shape change
void __check_shape(long long array_size, long long shape_size){
    if(array_size != shape_size){
        printf("the number of elements in array(%lld) does not match that of shape(%lld)\n",array_size,shape_size);
        assert(array_size == shape_size);
    }
}

void __check_shape(vector<int>& shape1, vector<int>& shape2){
    if(shape1.size() != shape2.size()){
        printf("operands could not be broadcast together with shapes\n");
        assert(shape1.size() == shape2.size());
    }
    for(int i=0;i<shape1.size();++i){
        if(shape1[i] != shape2[i]){
            printf("operands could not be broadcast together with shapes\n");
            assert(shape1[i] == shape2[i]);
        }
    }
}

void __check_index(int idx, int bound, int axis){
    if(idx >= bound || idx < 0){
        printf("index %d is out of bounds for axis %d with size %d\n",idx,axis,bound);
        assert(idx >= 0 && idx < bound);
    }
}

void __check_index(int idx, int bound){
    if(idx >= bound || idx < 0){
        printf("index %d is out of bounds with size %d\n",idx,bound);
        assert(idx >= 0 && idx < bound);
    }   
}

void __check_axes(vector<int>& axes, int ndim){
    try{
        string msg;
        if((int)axes.size() != ndim){
            msg = "axes don't match array";
            throw msg;
        }
        set<int> tmp;
        for(auto axis : axes){
            if(axis < -ndim || axis >= ndim){
                msg = "axis is out of bounds for array of dimension ";
                throw msg;
            }
            if(axis < 0){
                axis = ndim - axis;
            }
            if(tmp.find(axis) != tmp.end()){
                msg = "repeated axis in transpose";
                throw msg;
            }
            tmp.insert(axis);
        }
    }catch(const string msg){
        cout<<msg<<endl;
        assert(false);
    }
}

void __check_squeeze(vector<int> axis, vector<int> shape){
    for(auto i:axis){
        if(shape[i] > 1){
            printf("cannot select an axis to squeeze out which has size not equal to one\n");
            assert(false);
        }
    }
    
}