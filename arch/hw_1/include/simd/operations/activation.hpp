#pragma once

#include <cmath>
#include <type_traits>
#include <cstddef>
#include <stdexcept>
#include <xsimd/xsimd.hpp>

namespace simd::ops {

template<class T>
requires std::is_floating_point_v<T>
T silu_scalar(T x) {
    return x / (T{1} + std::exp(-x));
}

// Tanh approximation commonly used for GELU.
template<class T>
requires std::is_floating_point_v<T>
T gelu_scalar(T x) {
    constexpr T k = static_cast<T>(0.7978845608028654); // sqrt(2/pi)
    constexpr T c = static_cast<T>(0.044715);
    const T inner = k * (x + c * x * x * x);
    return T{0.5} * x * (T{1} + std::tanh(inner));
}

template<class T> requires std::is_floating_point_v<T>
void silu_simd(const T* input,T* output,std::size_t size) {
    if(size && (!input || !output)) throw std::invalid_argument("null array pointer");
    using batch=xsimd::batch<T>; constexpr std::size_t width=batch::size;
    std::size_t i=0;
    for(;i+width<=size;i+=width) { auto x=batch::load_unaligned(input+i); (x/(T{1}+xsimd::exp(-x))).store_unaligned(output+i); }
    for(;i<size;++i) output[i]=silu_scalar(input[i]);
}

template<class T> requires std::is_floating_point_v<T>
void gelu_simd(const T* input,T* output,std::size_t size) {
    if(size && (!input || !output)) throw std::invalid_argument("null array pointer");
    using batch=xsimd::batch<T>; constexpr std::size_t width=batch::size;
    const T k=static_cast<T>(0.7978845608028654), c=static_cast<T>(0.044715);
    std::size_t i=0;
    for(;i+width<=size;i+=width) { auto x=batch::load_unaligned(input+i); auto inner=k*(x+c*x*x*x); (T{0.5}*x*(T{1}+xsimd::tanh(inner))).store_unaligned(output+i); }
    for(;i<size;++i) output[i]=gelu_scalar(input[i]);
}

} // namespace simd::ops
