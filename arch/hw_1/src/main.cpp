#include <iostream>
#include <vector>

#include "simd/operations/arithmetic.hpp"
#include "simd/operations/reduction.hpp"
#include "simd/operations/activation.hpp"
#include "simd/matrix/matmul.hpp"

int main() {
    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{10, 20, 30, 40};
    std::vector<float> out(a.size());

    simd::ops::binary_transform_scalar(
        a.data(), b.data(), out.data(), a.size(), simd::ops::Add{});

    std::cout << "Scalar add: ";
    for (float x : out) std::cout << x << ' ';
    std::cout << "\nSum(a): " << simd::ops::sum_scalar(a.data(), a.size());
    std::cout << "\nMean(a): " << simd::ops::mean_scalar(a.data(), a.size());
    std::cout << "\nSiLU(1): " << simd::ops::silu_scalar(1.0f);
    std::cout << "\nGELU(1): " << simd::ops::gelu_scalar(1.0f) << '\n';
}
