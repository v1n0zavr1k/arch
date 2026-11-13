#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <xsimd/xsimd.hpp>

namespace simd::ops {
struct Add { template<class T> constexpr T operator()(T a,T b) const { return a+b; } };
struct Subtract { template<class T> constexpr T operator()(T a,T b) const { return a-b; } };
struct Multiply { template<class T> constexpr T operator()(T a,T b) const { return a*b; } };
struct Divide { template<class T> constexpr T operator()(T a,T b) const { if constexpr(std::is_integral_v<T>) if(b==0) throw std::domain_error("integer division by zero"); return a/b; } };

template<class T, class BinaryOp>
void binary_transform_scalar(const T* lhs,const T* rhs,T* output,std::size_t size,BinaryOp operation) {
    if(size && (!lhs || !rhs || !output)) throw std::invalid_argument("null array pointer");
    #pragma omp parallel for if(size > 32768)
    for(std::size_t i=0;i<size;++i) output[i]=operation(lhs[i],rhs[i]);
}

template<class T,class BinaryOp>
void binary_transform_simd(const T* lhs,const T* rhs,T* output,std::size_t size,BinaryOp operation) {
    if(size && (!lhs || !rhs || !output)) throw std::invalid_argument("null array pointer");
    using batch=xsimd::batch<T>; constexpr std::size_t width=batch::size;
    std::size_t i=0;
    #pragma omp parallel for if(size > 32768)
    for(std::ptrdiff_t block=0;block<static_cast<std::ptrdiff_t>(size/width);++block) {
        i=static_cast<std::size_t>(block)*width;
        const batch a=batch::load_unaligned(lhs+i), b=batch::load_unaligned(rhs+i);
        (operation(a,b)).store_unaligned(output+i);
    }
    i=(size/width)*width;
    for(;i<size;++i) output[i]=operation(lhs[i],rhs[i]);
}
} // namespace simd::ops
