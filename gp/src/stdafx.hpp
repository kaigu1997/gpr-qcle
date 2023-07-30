#ifndef STDAFX_HPP
#define STDAFX_HPP

#ifndef EIGEN_USE_MKL_ALL
#	define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>

#include <Eigen/Eigen>
#include <xtensor.hpp>
#include <pybind11/complex.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

template <std::size_t N, typename T>
	requires std::is_arithmetic_v<std::decay_t<T>>
inline constexpr T power([[maybe_unused]] const T t) noexcept
{
	if constexpr (N == 0)
	{
		return 1;
	}
	else if constexpr (N == 1)
	{
		return t;
	}
	else
	{
		return power<N / 2>(t) * power<N - N / 2>(t);
	}
}

constexpr std::size_t NUM_PES = 2, NUM_ELM = power<2>(NUM_PES), NUM_TRIG = NUM_PES * (NUM_PES + 1) / 2, DIM = 1, PHASEDIM = DIM * 2;
constexpr double mass = 2000.0, hbar = 1.0;

template <typename T>
using ClassicalVector = Eigen::Matrix<T, DIM, 1>;
using Matrix = Eigen::Matrix<double, NUM_PES, NUM_PES, Eigen::StorageOptions::RowMajor>;
using Vector = Eigen::Matrix<double, NUM_PES, 1>;
using range = std::ranges::iota_view<std::size_t, std::size_t>;
using DistributionFunction = std::function<std::complex<double>(const double, const double, const std::size_t, const std::size_t)>;

#endif // !STDAFX_HPP