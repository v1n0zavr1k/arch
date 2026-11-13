#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace simd::ops {

struct Add {
    template<class T>
    constexpr T operator()(T a, T b) const { return a + b; }
};

struct Subtract {
    template<class T>
    constexpr T operator()(T a, T b) const { return a - b; }
};

struct Multiply {
    template<class T>
    constexpr T operator()(T a, T b) const { return a * b; }
};

struct Divide {
    template<class T>
    constexpr T operator()(T a, T b) const {
        if constexpr (std::is_integral_v<T>) {
            if (b == 0) throw std::domain_error("integer division by zero");
        }
        return a / b;
    }
};

// Generic scalar kernel. The operation is injected as a callable object.
template<class T, class BinaryOp>
void binary_transform_scalar(
    const T* lhs, const T* rhs, T* output,
    std::size_t size, BinaryOp operation)
{
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = operation(lhs[i], rhs[i]);
    }
}

} // namespace simd::ops
