#ifndef FORCE_IMPORT_ARRAY
#	define FORCE_IMPORT_ARRAY
#endif // !FORCE_IMPORT_ARRAY

#ifndef EIGEN_USE_MKL_ALL
#	define EIGEN_USE_MKL_ALL
#endif // !EIGEN_USE_MKL_ALL

#ifdef DEF_ATTR
#	undef DEF_ATTR
#endif
#define DEF_ATTR(NAME) m.attr(#NAME) = NAME

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>

#include <Eigen/Eigen>
#include <xtensor.hpp>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
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

static constexpr std::size_t NUM_PES = 2, NUM_ELM = power<2>(NUM_PES), NUM_TRIG = NUM_PES * (NUM_PES + 1) / 2, DIM = 1, PHASEDIM = DIM * 2;
static constexpr double mass = 2000.0, hbar = 1.0;
static constexpr double DAC_A = 0.10, DAC_B = 0.28, DAC_C = 0.015, DAC_D = 0.06, DAC_E = 0.05;
static constexpr double dt = 0.1;
static constexpr std::size_t NumOffDiagonalBranches = 3;
static constexpr std::array<std::ptrdiff_t, NumOffDiagonalBranches> OffDiagonalBranches = {-1, 0, 1};
static constexpr std::size_t OffDiagonalZeroBranch = 1;

template <typename T>
using ClassicalVector = Eigen::Matrix<T, DIM, 1>;
using Matrix = Eigen::Matrix<double, NUM_PES, NUM_PES, Eigen::StorageOptions::RowMajor>;
using Vector = Eigen::Matrix<double, NUM_PES, 1>;
using range = std::ranges::iota_view<std::size_t, std::size_t>;
using OffDiagonalBranched = xt::xtensor_fixed<double, xt::xshape<NumOffDiagonalBranches>>;
using TotallyBranched = xt::xtensor_fixed<double, xt::xshape<NUM_TRIG, NumOffDiagonalBranches>>;

enum Direction
{
	Backward = -1,
	Forward = 1
};

static Matrix diabatic_potential(const double x) noexcept
{
	Matrix result = Matrix::Zero();
	result(1, 1) = DAC_E - DAC_A * std::exp(-DAC_B * power<2>(x));
	result(0, 1) = result(1, 0) = DAC_C * std::exp(-DAC_D * power<2>(x));
	return result;
}

static Matrix diabatic_force(const double x) noexcept
{
	Matrix result = Matrix::Zero();
	result(1, 1) = -2.0 * DAC_A * DAC_B * x * std::exp(-DAC_B * power<2>(x));
	result(0, 1) = result(1, 0) = 2.0 * DAC_C * DAC_D * x * std::exp(-DAC_D * power<2>(x));
	return result;
}

static Matrix diabatic_to_adiabatic_matrix(const double x) noexcept
{
	if constexpr (NUM_PES == 1)
	{
		// one level model
		return Matrix::Identity();
	}
	else if constexpr (NUM_PES == 2)
	{
		// 2-level diabatic
		const Matrix diaH = diabatic_potential(x);
		Matrix result;
		result.row(0) << -1.0, 1.0;
		result.row(0) *= std::sqrt(std::norm(diaH(0, 0) - diaH(1, 1)) + 4.0 * std::norm(diaH(0, 1)));
		result.row(0).array() += diaH(0, 0) - diaH(1, 1);
		result.row(0) /= 2.0 * diaH(1, 0);
		result.row(1).setOnes();
		result.array().rowwise() /= result.array().colwise().norm();
		return result;
	}
	else
	{
		// diabatic models with >=3 levels
		return Eigen::SelfAdjointEigenSolver<Matrix>(diabatic_potential(x)).eigenvectors();
	}
}

static Vector adiabatic_potential(const double x) noexcept
{
	if constexpr (NUM_PES == 1)
	{
		// one level model
		return diabatic_potential(x);
	}
	else if constexpr (NUM_PES == 2)
	{
		// 2-level diabatic
		const Matrix diaH = diabatic_potential(x);
		Vector result;
		result << -1.0, 1.0;
		result *= std::sqrt(std::norm(diaH(0, 0) - diaH(1, 1)) + std::norm(2.0 * diaH(1, 0)));
		result.array() += diaH(0, 0) + diaH(1, 1);
		result /= 2.0;
		return result;
	}
	else
	{
		// diabatic models with >=3 levels
		return Eigen::SelfAdjointEigenSolver<Matrix>(diabatic_potential(x)).eigenvalues();
	}
}

static Matrix adiabatic_force(const double x) noexcept
{
	const Matrix ts_mat = diabatic_to_adiabatic_matrix(x);
	Matrix result = ts_mat.adjoint() * diabatic_force(x) * ts_mat;
	return (result + result.adjoint()) / 2.0;
}

static Matrix adiabatic_coupling(const double x) noexcept
{
	const Vector E = adiabatic_potential(x);
	const Matrix F = adiabatic_force(x);
	Matrix result = Matrix::Zero();
	for (const std::size_t iPES: range{1, NUM_PES})
	{
		for (const std::size_t jPES: range{0, iPES})
		{
			result(iPES, jPES) = F(iPES, jPES) / (E[iPES] - E[jPES]);
			result(jPES, iPES) = -result(iPES, jPES);
		}
	}
	return result;
}

static bool is_coupling([[maybe_unused]] const double x, [[maybe_unused]] const double p)
{
	[[maybe_unused]] static constexpr double CouplingCriterion = 0; // Criteria of whether have coupling or not
	static constexpr bool IsAdiabatic = false;
	if constexpr (IsAdiabatic)
	{
		return false;
	}
	else
	{
		const Matrix Force = adiabatic_force(x), NAC = adiabatic_coupling(x);
		const double diag_f_average = Force.diagonal().mean();
		if constexpr (NUM_PES == 2)
		{
			return std::abs(NAC(0, 1) * p / mass) * dt >= CouplingCriterion
				|| std::abs(Force(0, 1) / diag_f_average) >= CouplingCriterion;
		}
		else
		{
			for (const std::size_t iPES : range{1ul, NUM_PES})
			{
				for (const std::size_t jPES : range{0, iPES})
				{
					if (std::abs(NAC(iPES, jPES) * p / mass) > CouplingCriterion
						|| std::abs(Force(iPES, jPES) / diag_f_average) > CouplingCriterion)
					{
						return true;
					}
				}
			}
			return false;
		}
	}
}

static std::tuple<double, double> adiabatic_evolve(
	double x,
	double p,
	const double dt,
	const Direction drc,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	auto position_evolve = [&x, p, dt, drc](void) -> void
	{
		x += static_cast<int>(drc) * dt / 2.0 * p / mass;
	};
	auto momentum_diagonal_nonbranch_evolve = [x, &p, dt, drc, RowIndex, ColIndex](void) -> void
	{
		const Matrix f = adiabatic_force(x);
		p += static_cast<int>(drc) * dt / 2.0 * (f(RowIndex, RowIndex) + f(ColIndex, ColIndex));
	};
	position_evolve();
	momentum_diagonal_nonbranch_evolve();
	position_evolve();
	return std::make_tuple(x, p);
}

static inline double calculate_omega0(
	const double x0,
	const double x2,
	const Direction drc,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	if (RowIndex == ColIndex)
	{
		return 0.0;
	}
	const Vector E0 = adiabatic_potential(x0), E2 = adiabatic_potential(x2);
	return static_cast<int>(drc) * (E0[RowIndex] - E0[ColIndex] + E2[RowIndex] - E2[ColIndex]) / 2.0 / hbar;
}

std::complex<double> non_adiabatic_evolve_predict(
	const double x0,
	const double p0,
	const std::optional<std::complex<double>> density,
	const std::function<std::complex<double>(const xt::pyarray<double>&, const std::size_t, const std::size_t)>& distribution,
	const std::size_t RowIndex,
	const std::size_t ColIndex
)
{
	using namespace std::literals::complex_literals;
	static constexpr Direction drc = Direction::Backward;
	static auto calculate_lower_triangular_index = [](const std::size_t Row, const std::size_t Col) constexpr -> std::size_t
	{
		return Row * (Row + 1) / 2 + Col;
	};
	auto position_evolve =
		[]<std::size_t... xIndices, std::size_t... pIndices>(
			const xt::xtensor_fixed<double, xt::xshape<xIndices...>>& x,
			const xt::xtensor_fixed<double, xt::xshape<pIndices...>>& p,
			const double dt
		)
			->xt::xtensor_fixed<double, xt::xshape<pIndices...>>
	{
		return x + static_cast<int>(drc) * dt * p / mass;
	};
	const bool IsCouple = is_coupling(x0, p0);
	auto offdiagonal_rotation =
		[IsCouple](auto rho_view, const double x, const double p, const double dt) -> void
	{
		// rhoview are expected to have 3 elements
		assert(rho_view.shape()[0] == NUM_TRIG && rho_view.dimension() == 1);
		// phi: v.dot(NAC)
		const double phi = is_coupling(x, p) ? p / mass * adiabatic_coupling(x)(0, 1) : 0.0;
		const double cosphi = std::cos(2.0 * phi * dt), sinphi = std::sin(2.0 * phi * dt);
		// save the old data
		const xt::xtensor_fixed<std::complex<double>, xt::xshape<NUM_TRIG>> rho_save = rho_view;
		rho_view(0) = (1.0 + cosphi) / 2.0 * rho_save(0) - sinphi * rho_save(1).real() + (1.0 - cosphi) / 2.0 * rho_save(2);
		rho_view(1) = sinphi / 2.0 * rho_save(0) + cosphi * rho_save(1).real() + 1.0i * rho_save(1).imag() - sinphi / 2.0 * rho_save(2);
		rho_view(2) = (1.0 - cosphi) / 2.0 * rho_save(0) + sinphi * rho_save(1).real() + (1.0 + cosphi) / 2.0 * rho_save(2);
	};
	// x_i and p_{i-1} have same number of elements, and p_i is branching
	// 17 steps
	// index: {classical dimensions}
	const auto [x2, p1] = adiabatic_evolve(x0, p0, dt / 2.0, drc, RowIndex, ColIndex);
	// index: {offdiagonal branching, classical dimensions}
	// (backward) direction is included. So when evolve forward, the branch correspondence remains the same
	const auto p2 = [x2, p1, IsCouple](void) -> OffDiagonalBranched
	{
		const double f01 = IsCouple ? adiabatic_force(x2)(0, 1) : 0.0;
		const auto n_broadcast = xt::view(xt::adapt(OffDiagonalBranches.data(), xt::xshape<NumOffDiagonalBranches>{}), xt::all());
		return p1 + dt * n_broadcast * f01;
	}();
	const OffDiagonalBranched x3 = position_evolve(xt::xtensor_fixed<double, xt::xshape<1>>(x2 * xt::ones<double>(xt::xshape<1>{})), p2, dt / 4.0);
	// index: {classical dimensions, density matrix element it goes to, offdiagonal branching, density matrix element it comes from}
	const TotallyBranched p3 = [&x3, &p2, dt = dt / 2.0](void)
	{
		TotallyBranched result;
		// recursively reduce rank, from highest rank to lowest
		for (const std::size_t iOffDiagonalBranch : range{0, NumOffDiagonalBranches})
		{
			const double xview = x3[iOffDiagonalBranch], pview = p2[iOffDiagonalBranch];
			const Matrix f = adiabatic_force(xview);
			for (const std::size_t iPES : range{0, NUM_PES})
			{
				for (const std::size_t jPES : range{0, iPES + 1})
				{
					result(calculate_lower_triangular_index(iPES, jPES), iOffDiagonalBranch) =
						pview + static_cast<int>(drc) * dt / 2.0 * (f(iPES, iPES) + f(jPES, jPES));
				}
			}
		}
		return result;
	}();
	const TotallyBranched x4 = position_evolve(x3, p3, dt / 4.0);
	// rho(gamma', n, gamma) = rho^{gamma'}_W(R4(gamma', n, gamma, .), P3(gamma', n, gamma, .), t)
	// The predicted density at all those points
	auto rho_predict = [&x4, &p3, &density, &distribution, RowIndex, ColIndex](void)
	{
		xt::xtensor_fixed<std::complex<double>, xt::xshape<NUM_TRIG, NumOffDiagonalBranches>> result;
		for (const std::size_t iPES : range{0, NUM_PES})
		{
			for (const std::size_t jPES : range{0, iPES + 1})
			{
				const std::size_t TrigIndex = calculate_lower_triangular_index(iPES, jPES);
				// construct training set
				const auto r = [&x4, &p3, TrigIndex](void) -> Eigen::Matrix<double, PHASEDIM, NumOffDiagonalBranches>
				{
					const auto xview = xt::view(x4, TrigIndex, xt::all()), pview = xt::view(p3, TrigIndex, xt::all());
					Eigen::Matrix<double, PHASEDIM, NumOffDiagonalBranches> result;
					result.block<DIM, NumOffDiagonalBranches>(0, 0) =
						Eigen::Matrix<double, DIM, NumOffDiagonalBranches>::Map(xview.data() + xview.data_offset());
					result.block<DIM, NumOffDiagonalBranches>(DIM, 0) =
						Eigen::Matrix<double, DIM, NumOffDiagonalBranches>::Map(pview.data() + pview.data_offset());
					return result;
				}();
				// get prediction
				for (const std::size_t iOffDiagonalBranch : range{0, NumOffDiagonalBranches})
				{
					if (iPES == RowIndex && jPES == ColIndex && OffDiagonalBranches[iOffDiagonalBranch] == 0 && density.has_value())
					{
						// the exact element, assign the exact density
						result(TrigIndex, iOffDiagonalBranch) = density.value();
					}
					else
					{
						result(TrigIndex, iOffDiagonalBranch) = distribution(xt::pyarray<double>(xt::adapt(r.col(iOffDiagonalBranch).data(), xt::xshape<PHASEDIM>{})), iPES, jPES);
					}
				}
			}
		}
		return result;
	}();
	xt::xtensor_fixed<std::complex<double>, xt::xshape<NUM_TRIG>> rho_combined_offdiag = xt::zeros<decltype(rho_combined_offdiag)::value_type>(decltype(rho_combined_offdiag)::shape_type{});
	for (const std::size_t iOffDiagonalBranch : range{0, NumOffDiagonalBranches})
	{
		// first, an adiabatic evolve. x4 and p3 goes to x2 and p2, rho evolves in the following way
		const double x4view = x4(calculate_lower_triangular_index(1, 0), iOffDiagonalBranch);
		rho_predict(calculate_lower_triangular_index(1, 0), iOffDiagonalBranch) *=
			std::exp(calculate_omega0(x2, x4view, Direction::Forward, 0, 1) * dt / 2 * 1.0i);
		// now they are at x2 and p2. A off-diagonal rotation is needed.
		const double p2view = p2[iOffDiagonalBranch];
		offdiagonal_rotation(
			xt::view(rho_predict, xt::all(), iOffDiagonalBranch),
			x2,
			p2view,
			dt / 2.0
		);
		// then the off-diagonal force evolution combination with a rotation matrix, p2 comes to p1, x2 keeps the same
		switch (OffDiagonalBranches[iOffDiagonalBranch])
		{
		case -1:
			rho_combined_offdiag +=
				(rho_predict(0, iOffDiagonalBranch) + 2.0 * rho_predict(1, iOffDiagonalBranch).real() + rho_predict(2, iOffDiagonalBranch)) / 4.0;
			break;
		case 0:
		{
			const std::complex<double> value = (rho_predict(0, iOffDiagonalBranch) - rho_predict(2, iOffDiagonalBranch)) / 2.0;
			rho_combined_offdiag(0) += value;
			rho_combined_offdiag(1) += 1.0i * rho_predict(1, iOffDiagonalBranch).imag();
			rho_combined_offdiag(2) -= value;
			break;
		}
		case 1:
		{
			const std::complex<double> value =
				(rho_predict(0, iOffDiagonalBranch) - 2.0 * rho_predict(1, iOffDiagonalBranch).real() + rho_predict(2, iOffDiagonalBranch)) / 4.0;
			rho_combined_offdiag(0) += value;
			rho_combined_offdiag(1) -= value;
			rho_combined_offdiag(2) += value;
			break;
		}
		[[unlikely]] default:
			assert(!"WRONG BRANCH!");
			break;
		}
	}
	// the another off-diagonal rotation at x2 and p1
	offdiagonal_rotation(
		xt::view(rho_combined_offdiag, xt::all()),
		x2,
		p1,
		dt / 2.0
	);
	// now another adiabatic evolve from (x2, p1) to (x0, p0), i.e., r
	const std::complex<double> result = rho_combined_offdiag(calculate_lower_triangular_index(RowIndex, ColIndex));
	if (RowIndex != ColIndex)
	{
		return result * std::exp(calculate_omega0(x0, x2, Direction::Forward, 0, 1) * dt / 2.0 * 1.0i);
	}
	else
	{
		return result;
	}
}

PYBIND11_MODULE(evolve, m)
{
	using namespace py::literals;
	xt::import_numpy();

	DEF_ATTR(NUM_PES);
	DEF_ATTR(NUM_ELM);
	DEF_ATTR(NUM_TRIG);
	DEF_ATTR(PHASEDIM);
	m.def("non_adiabatic_evolve_predict", non_adiabatic_evolve_predict, "x0"_a, "p0"_a, "density"_a, "distribution"_a, "RowIndex"_a, "ColIndex"_a);
}

#undef DEF_ATTR
#undef DEF_FUNC
