#pragma once

#include <cmath>
#include <type_traits>

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

} // namespace simd::ops
