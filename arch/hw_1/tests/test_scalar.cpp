#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include "simd/operations/arithmetic.hpp"
#include "simd/operations/reduction.hpp"
#include "simd/operations/activation.hpp"
#include "simd/matrix/matmul.hpp"

TEST(ScalarArithmetic, AddFloat) {
    const std::array<float, 3> a{1.f, 2.f, 3.f};
    const std::array<float, 3> b{4.f, 5.f, 6.f};
    std::array<float, 3> out{};
    simd::ops::binary_transform_scalar(a.data(), b.data(), out.data(), a.size(), simd::ops::Add{});
    EXPECT_FLOAT_EQ(out[0], 5.f);
    EXPECT_FLOAT_EQ(out[1], 7.f);
    EXPECT_FLOAT_EQ(out[2], 9.f);
}

TEST(ScalarArithmetic, IntegerDivisionByZeroThrows) {
    EXPECT_THROW((simd::ops::Divide{}(10, 0)), std::domain_error);
}

TEST(ScalarReduction, SumAndMean) {
    const std::array<std::int32_t, 4> values{1, 2, 3, 4};
    EXPECT_EQ(simd::ops::sum_scalar(values.data(), values.size()), 10);
    EXPECT_DOUBLE_EQ(simd::ops::mean_scalar(values.data(), values.size()), 2.5);
}

TEST(ScalarReduction, EmptyMeanThrows) {
    const std::array<double, 0> values{};
    EXPECT_THROW((simd::ops::mean_scalar(values.data(), values.size())), std::invalid_argument);
}

TEST(ScalarActivation, SiluAndGeluAtZero) {
    EXPECT_NEAR(simd::ops::silu_scalar(0.0), 0.0, 1e-12);
    EXPECT_NEAR(simd::ops::gelu_scalar(0.0), 0.0, 1e-12);
}

TEST(ScalarMatmul, MultipliesRowMajorMatrices) {
    const std::array<double, 4> a{1, 2, 3, 4};
    const std::array<double, 4> b{5, 6, 7, 8};
    std::array<double, 4> c{};
    simd::matrix::matmul_scalar(a.data(), b.data(), c.data(), 2, 2, 2);
    EXPECT_DOUBLE_EQ(c[0], 19);
    EXPECT_DOUBLE_EQ(c[1], 22);
    EXPECT_DOUBLE_EQ(c[2], 43);
    EXPECT_DOUBLE_EQ(c[3], 50);
}
