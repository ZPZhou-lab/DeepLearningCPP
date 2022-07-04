#include <bits/stdc++.h>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <vector>
#include <typeinfo>
#include <stdarg.h>
using namespace std;


void __check_shape(const long long &array_size, const long long &shape_size){
    if(array_size != shape_size){
        printf("the number of elements in array(%lld) does not match that of shape(%lld)\n",array_size,shape_size);
        assert(array_size == shape_size);
    }
}

void __check_shape(const vector<int>& shape1, const vector<int>& shape2){
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

void __check_index(const int &idx, const int &bound, const int &axis){
    if(idx >= bound || idx < 0){
        printf("index %d is out of bounds for axis %d with size %d\n",idx,axis,bound);
        assert(idx >= 0 && idx < bound);
    }
}

void __check_index(const int &idx, const int &bound){
    if(idx >= bound || idx < 0){
        printf("index %d is out of bounds with size %d\n",idx,bound);
        assert(idx >= 0 && idx < bound);
    }   
}

void __check_axes(const vector<int>& axes, const int &ndim){
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

void __check_squeeze(const vector<int> &axis, const vector<int> &shape){
    for(auto i:axis){
        if(shape[i] > 1){
            printf("cannot select an axis to squeeze out which has size not equal to one\n");
            assert(false);
        }
    }
    
}

void __check_axis(const int &ndim, const int &axis){
    if(axis >= ndim){
        printf("axis %d is out of bounds for array of dimension %d\n",axis,ndim);
        assert(false);
    }
}

void __check_repeat(const int &repeats){
    if(repeats < 1){
        printf("non-positive dimensions are not allowed\n");
        assert(false);
    }
}

void __check_expand(const int ndim, const vector<int> &axis){
    int new_ndim = ndim + axis.size();
    for(auto a:axis){
        if(a >= new_ndim){
            printf("axis %d is out of bounds for array of dimension %d\n",a,new_ndim);
            assert(false);
        }
    }
}

void __check_dot(const long long size1, const long long size2){
    if(size1 != size2){
        printf("shapes (%lld,) and (%lld,) not aligned: %lld (dim 0) != %lld (dim 0)\n",size1,size2,size1,size2);
        assert(false);
    }
}