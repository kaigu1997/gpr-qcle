r"""wendland
========
The module for Wendland kernel functions
"""
import collections.abc
import math
import os
import typing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import numpy.typing as npt
import scipy.integrate
import sympy as sp
import torch

import constant

# if constant.DEBUG_MODE:
# 	torch.sparse.check_sparse_tensor_invariants.enable()
# else:
# 	torch.sparse.check_sparse_tensor_invariants.disable()

_EPS: typing.Final = math.sqrt(torch.finfo(torch.float).eps * torch.finfo(torch.double).eps)


class cdist2:
	r"""To calculate the pairwise distance in a numerically stable way

	Parameters
	----------
	x1 : torch.Tensor
		Input tensor where the last two dimensions represent the points and the feature dimension respectively.
		The shape can be :math:`D_1 \times D_2 \times \cdots \times D_n \times P \times M`,
		where :math:`P` is the number of points and :math:`M` is the feature dimension.
	x2 : torch.Tensor
		Input tensor where the last two dimensions also represent the points and the feature dimension respectively.
		The shape can be :math:`D_1' \times D_2' \times \cdots \times D_m' \times R \times M`,
		where :math:`R` is the number of points and :math:`M` is the feature dimension,
		which should match the feature dimension of `x1`.
	eps : float, optional
		The precision under which the distance is not calculated and clamped, by default _EPS

	Returns
	-------
	torch.Tensor
		If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
		output will have shape :math:`B \times P \times R`.

	Notes
	-----
	For memory efficiency, the squared distance is calculated by :math:`||x_1 - x_2||^2=||x_1||^2+||x_2||^2-2x_1^T x_2`.

	For numerical stability, the squared distance is clamped to eps.
	"""
	__EPS: typing.Final = torch.finfo(torch.get_default_dtype()).eps ** 2
	def __new__(cls, x1: torch.Tensor, x2: torch.Tensor, eps: float = __EPS) -> torch.Tensor:
		return ((x1 ** 2).sum(dim=-1, keepdim=True) + (x2 ** 2).sum(dim=-1, keepdim=True).mT - 2 * x1 @ x2.mT).clamp(min=eps)


@typing.final
class _wendland:
	r"""To calculate the Wendland polynomial

	Parameters
	----------
	dim : int
		The dimension of the input features
	k : int, optional
		The smoothness parameter, by default 1

	Notes
	-----
	The Wendland functions are defined as follows:

	$$\phi_{d,k}(r) = \begin{cases}
	(1 - r)_+^{2k + \lfloor \frac{d}{2} \rfloor + 1} P_{d,k}(r), & 0 \leq r < 1 \\
	0, & r \geq 1
	\end{cases}$$

	where $P_{d,k}(r)$ is a polynomial of degree $k$ that ensures the desired smoothness.
	"""
	@typing.final
	class __LazyList[T]:
		r"""A list that extends itself on out-of-bound access via a builder function.

		Parameters
		----------
		builder : collections.abc.Callable[[int], T]
			Builder function that takes the index and returns the value to be appended when the index is out of bound
		initial : list[T], optional
			The initial values of the list, by default []
		"""
		__slots__: typing.Final[tuple] = ("_items", "_builder")
		_items: list[T]
		_builder: collections.abc.Callable[[int, list[T]], T]

		def __init__(self, builder: collections.abc.Callable[[int, list[T]], T], initial: list[T] = []) -> None:
			self._items = initial
			self._builder = builder

		def __getitem__(self, index: int) -> T:
			r"""To get the item at the given index, and extend the list if the index is out of bound

			Parameters
			----------
			index : int
				The index of the item to be accessed

			Returns
			-------
			T
				The item at the given index
			"""
			if len(self._items) <= index:
				for i in range(len(self._items), index + 1):
					self._items.append(self._builder(i, self._items))
			return self._items[index]

	@staticmethod
	def __next_builder(i: int, items: list[list[int]]) -> list[int]:
		r"""The builder function for the lazy list of Wendland coefficients

		This function should be called sequentially with increasing index, and it will return the list of Wendland coefficients for the next index based on the previous one.

		Parameters
		----------
		i : int
			The next index of the coefficient to be built

		Returns
		-------
		list[int]
			The list of Wendland coefficients for the next index
		"""
		if i == 0:
			return [1]
		else:
			prev = items[i-1]
			new = [i * prev[-1]]
			for j in range(i-2, -1, -1):
				new.append(new[-1] + (j+1)*prev[j])
			new.append(new[-1])
			new.reverse()
			return new

	@typing.final
	class __classproperty[T]:
		r"""To define a class property that can be accessed without instantiating the class, and can be used in classmethod and staticmethod

		Parameters
		----------
		func : collections.abc.Callable[..., T]
			The function to be called when the property is accessed, which takes the class as the first argument and returns the value of the property
		"""
		__slots__: typing.Final[tuple] = ("_func",)
		_func: collections.abc.Callable[..., T]

		def __init__(self, func: collections.abc.Callable[..., T]):
			self._func = func

		def __get__(self, obj: object, cls: type | None = None) -> T:
			r"""To get the value of the class property

			Parameters
			----------
			obj : object
				The instance of the class, which is not used in this case
			cls : type | None, optional
				The class to which the property belongs, by default None

			Returns
			-------
			T
				The value of the class property
			"""
			return self._func(cls or type(obj))

	@__classproperty
	def B_kn(cls) -> __LazyList[list[int]]:
		return cls.__B_kn_table

	__B_kn_table: typing.Final[__LazyList[list[int]]] = __LazyList(__next_builder)
	__slots__: typing.Final[tuple] = ("_k", "_power_1_r", "_gamma")
	_k: typing.Final[int]
	_power_1_r: typing.Final[int]
	_gamma: torch.Tensor

	def __init__(
		self,
		dim: int,
		k: int = 1,
	) -> None:
		m: typing.Final[int] = dim // 2
		self._k = k
		l: typing.Final[int] = m + self._k + 1
		degree = l + 2 * k
		self._power_1_r = l + k
		denom: typing.Final[sp.Integer] = typing.cast(sp.Integer, sp.factorial2(2*k-1))
		# (1-r)^{l+k} * sum_{n=0}^k gamma_{l,k,n} * r^n
		gamma_sp = [sp.Rational((-1) ** n * sp.Add(*(sp.Mul((-1) ** j, _wendland.B_kn[k][j], sp.binomial(degree, j), sp.binomial(k-j, n-j)) for j in range(n + 1))), denom) for n in range(k + 1)]
		self._gamma = torch.tensor([float(a) for a in gamma_sp], device="cpu") # this will be moved to GPU later, after chebychev grids are constructed

	@property
	def k(self) -> int:
		r"""The smoothness parameter

		Returns
		-------
		int
			The smoothness parameter
		"""
		return self._k

	@typing.overload
	def __call__(self, r: torch.Tensor) -> torch.Tensor: ...
	@typing.overload
	def __call__(self, r: float) -> float: ...

	def __call__(self, r: torch.Tensor | float) -> torch.Tensor | float:
		r"""To calculate the Wendland polynomial, which is the part of Wendland function that is not the exponential decay

		Parameters
		----------
		r : torch.Tensor or float
			The distance between two points divided by the cutoff radius, which is the input of the Wendland function

		Returns
		-------
		torch.Tensor or float
			The value of the Wendland polynomial, which is the part of Wendland function that is not the exponential decay

		Notes
		-----
		Autograd is not supported for csr and csc format, on either CPU or CUDA.
		"""
		result: torch.Tensor | float
		if isinstance(r, torch.Tensor):
			if self._k >= 8: # Estrin's scheme:
				result = self._gamma[:, *([torch.newaxis] * r.ndim)] * torch.ones_like(r)
				n = self._k
				value_2_pow = r
				while n > 0:
					for i in range(0, n, 2):
						result[i//2] = result[i] + result[i+1] * value_2_pow
					if n % 2 == 0:
						result[n//2] = result[n]
					value_2_pow = value_2_pow * value_2_pow
					n //= 2
				result = result[0]
			else:
				result = torch.full_like(r, self._gamma[self._k].item())
				for i in range(self._k - 1, -1, -1):
					result = result * r + self._gamma[i]
			return result * (1 - r) ** self._power_1_r
		else: # isinstance(r, float):
			result = self._gamma[self._k].item()
			for i in range(self._k - 1, -1, -1):
				result = r * result + self._gamma[i].item()
			return (1 - r) ** self._power_1_r * result

	def to(self, device: torch.device) -> None:
		self._gamma = self._gamma.to(device)


class wendland_rbf:
	r"""To calculate the covariance matrix using Wendland x rbf kernel

	Parameters
	----------
	dim : int
		The dimension of the input features
	k : int, optional
		The smoothness parameter, by default 1
	chebyshev_N : int, optional
		The number of Chebyshev grids for fitting the square integral, by default __chebyshev_N
	max_support_radius : float, optional
		The maximum support radius, by default __max_support_radius
	eps : float, optional
		The precision of the numerical integration on Chebyshev grids, by default _EPS
	"""

	@typing.final
	class __LazyDict[T]:
		r"""A dictionary that extends itself on out-of-bound access via a builder function.

		Parameters
		----------
		builder : collections.abc.Callable[[int], T]
			Builder function that takes the index and returns the value to be appended when the index is out of bound
		initial : dict[int, T], optional
			The initial values of the dictionary, by default {}
		"""
		__slots__: typing.Final[tuple] = ("__items", "__builder")
		__items: dict[int, T]
		__builder: collections.abc.Callable[[int], T]

		def __init__(self, builder: collections.abc.Callable[[int], T], initial: dict[int, T] = {}) -> None:
			self.__items = initial
			self.__builder = builder

		def __getitem__(self, index: int) -> T:
			r"""To get the item at the given index, and extend the dictionary if the index is out of bound

			Parameters
			----------
			index : int
				The index of the item to be accessed

			Returns
			-------
			T
				The item at the given index
			"""
			if index not in self.__items:
				self.__items[index] = self.__builder(index)
			return self.__items[index]

	__SQRT_PI_2: typing.Final = math.sqrt(math.pi / 2.0)
	__SQRT_1_2: typing.Final = math.sqrt(1.0 / 2.0)
	r_c_init: typing.Final = 2.5
	max_support: typing.Final = 4.0
	__r_c: typing.Final[sp.Symbol] = sp.Symbol("r_c", positive=True)
	__chebyshev_N: typing.Final = 64
	__grid_close_eps: typing.Final = torch.finfo(torch.get_default_dtype()).eps
	__slots__: typing.Final[tuple] = ("__poly", "__degree_range", "__even_ub", "__odd_ub", "__alpha_sp", "__a1", "__j_coe", "__integrals", "__rc_values", "__rc_chebyshev_grids", "__rc_chebyshev_func_vals", "__rc_chebyshev_weights")
	__poly: typing.Final[_wendland]
	__degree_range: typing.Final[npt.NDArray[np.int_]]
	__even_ub: typing.Final[int]
	__odd_ub: typing.Final[int]
	__alpha_sp: typing.Final[list[sp.Rational]]
	__a1: typing.Final[list[float]]
	__j_coe: typing.Final[list[float]]
	__integrals: typing.Final[__LazyDict[sp.Expr]]
	__rc_values: typing.Final[torch.Tensor]
	__rc_chebyshev_grids: typing.Final[torch.Tensor]
	__rc_chebyshev_func_vals: typing.Final[torch.Tensor]
	__rc_chebyshev_weights: typing.Final[torch.Tensor]

	def __symbol_integral(self, d: int) -> sp.Expr:
		r"""To calculate the integral of :math: \int_0^{r_c} \phi_{d,k}(r) r^{\mathtt{extra\_dim}} dr

		Parameters
		----------
		d : int
			The extra dimension of the integral, which is the power of r

		Returns
		-------
		sp.Expr
			The value of the integral in sympy expression
		"""
		return sp.Add(*(sp.Mul(sp.Integer(2) ** sp.Rational(t + d - 1, sp.Integer(2)), sp.lowergamma(sp.Rational(t + d + 1, sp.Integer(2)), sp.Mul(wendland_rbf.__r_c**2, sp.Rational(1, 2))), self.__alpha_sp[t], wendland_rbf.__r_c**-t) for t in self.__degree_range)).simplify()

	def __init__(
		self,
		dim: int,
		k: int = 1,
		chebyshev_N: int = __chebyshev_N,
		max_support_radius: float = max_support,
		eps: float = _EPS
	) -> None:
		poly: typing.Final = _wendland(dim, k)
		self.__poly = poly
		m: typing.Final[int] = dim // 2
		l: typing.Final[int] = m + k + 1
		degree = l + 2 * k
		self.__degree_range = np.arange(degree + 1)
		self.__even_ub = degree // 2
		self.__odd_ub = (degree - 1) // 2
		denom: typing.Final[sp.Integer] = typing.cast(sp.Integer, sp.factorial2(2*k-1))
		# sum_{n=0}^{l+2k} alpha_{l,k,j} * r^j
		self.__alpha_sp = [typing.cast(sp.Rational, sp.Mul(sp.Add(*(sp.Mul((-1) ** n, sp.binomial(t, n), _wendland.B_kn[k][n]) for n in range(min(k, t) + 1))), (-1) ** t, sp.Rational(sp.binomial(degree, t), denom)).doit()) for t in self.__degree_range]
		self.__a1 = [float(typing.cast(float, sp.Mul(sp.factorial2(t + dim - 2), self.__alpha_sp[t]).evalf())) for t in range(degree + 1)]
		self.__j_coe = [0] * (k+1) + [float(typing.cast(float, sp.Add(*(sp.Mul(
			sp.Rational(sp.factorial2(t + dim - 2), sp.factorial2(t - 2 * j)),
			self.__alpha_sp[t]
		) for t in range(2*j, degree + 1))).evalf())) for j in range(k + 1, self.__even_ub + 1)]
		self.__integrals = wendland_rbf.__LazyDict(self.__symbol_integral)
		# sum_{n=0}^{j} alpha_{l,k,n} * alpha_{l,k,j-n}
		alpha2_sp: typing.Final[list[sp.Rational]] = [typing.cast(sp.Rational, sp.Add(*(sp.Mul(self.__alpha_sp[n], self.__alpha_sp[j-n]) for n in range(max(0, j-degree),min(j, degree) + 1))).doit().simplify()) for j in range(0, degree * 2 + 1)]
		integral2: typing.Final[sp.Expr] = sp.Mul(sp.pi ** m, sp.Rational(1.0, sp.factorial(m)), sp.Add(*(sp.Mul(alpha2_sp[j], wendland_rbf.__r_c**(-j), sp.lowergamma(sp.Rational(j + dim, sp.Integer(2)), wendland_rbf.__r_c**2)) for j in range(degree * 2 + 1)))).simplify()
		# numerical solution
		self.__rc_values = (torch.cos(torch.arange(chebyshev_N + 1) * math.pi / chebyshev_N) + 1) / 2 * max_support_radius # range max_support to 0
		self.__rc_chebyshev_grids = torch.cos(torch.arange(chebyshev_N + 1) * math.pi / chebyshev_N) + 1
		self.__rc_chebyshev_func_vals = torch.empty((chebyshev_N + 1, chebyshev_N + 1))
		self.__rc_chebyshev_func_vals[:, 0] = 0 # cutoff distance
		self.__rc_chebyshev_func_vals[:-1, -1] = torch.tensor([integral2.evalf(subs={wendland_rbf.__r_c: r.item()}) for r in self.__rc_values[:-1]], dtype=torch.get_default_dtype(), device=torch.get_default_device())
		self.__rc_chebyshev_func_vals[-1, :] = 0 # r_c = 0
		prefactor: typing.Final[float] = 2 * (2 * math.pi) ** (m - 1) / int(typing.cast(int, sp.factorial2(dim - 3)))
		self.__rc_chebyshev_func_vals[:-1, 1:-1] = torch.tensor(
			joblib.Parallel(
				min(chebyshev_N, joblib.cpu_count()),
				verbose=50 if constant.DEBUG_MODE else 0
			)(
				joblib.delayed(lambda dist, r_c: prefactor * r_c ** dim * math.exp(-r_c**2 * dist ** 2 / 4) * scipy.integrate.dblquad(lambda w, v: math.exp(-r_c**2 * (w**2 + v**2)) * poly(math.sqrt((w - dist / 2) ** 2 + v ** 2)) * poly(math.sqrt((w + dist / 2) ** 2 + v ** 2)) * v ** (dim - 2), 0, math.sqrt(1 - dist**2 / 4), lambda v: dist / 2 - math.sqrt(1 - v**2), lambda v: math.sqrt(1 - v**2) - dist / 2, (), eps, eps)[0])(dist, r_c)
				for r_c in self.__rc_values[:-1].tolist() for dist in self.__rc_chebyshev_grids[1:-1].tolist()
			),
			dtype=torch.get_default_dtype(),
			device=torch.get_default_device()
		).reshape(chebyshev_N, chebyshev_N - 1) # total N+1 grids, head and tail are not included
		self.__poly.to(torch.get_default_device())
		self.__rc_chebyshev_weights = ((-1) ** torch.arange(chebyshev_N + 1)).to(torch.get_default_dtype())
		self.__rc_chebyshev_weights[[0, -1]] /= 2

	def __call__(
		self,
		x1: torch.Tensor,
		x2: torch.Tensor | None = None,
		/, *,
		lengthscale: torch.Tensor,
		r_c: torch.Tensor
	) -> torch.Tensor:
		r"""To calculate the covariance matrix

		Parameters
		----------
		x1 : torch.Tensor, shape of (...., M, D)
			The first feature set
		x2 : torch.Tensor, shape of (..., N, D)
			The second feature set
		lengthscale : torch.Tensor, shape of (D,)
			The characteristic lengths
		r_c : torch.Tensor
			The support radius

		Returns
		-------
		torch.Tensor, shape of (..., M, N)
			The covariance matrix, in COO format
		"""
		assert x1.shape[-1] == lengthscale.numel() and x1.ndim >= 2
		lengthscale = lengthscale.reshape(-1)
		x1 = x1 / lengthscale
		dist2: torch.Tensor
		mask: torch.Tensor
		if x2 is not None:
			assert x2.shape[-1] == lengthscale.numel() and x2.ndim >= 2
			x2 = x2 / lengthscale
			dist2 = cdist2(x1, x2)
			mask = dist2 < r_c ** 2
		else:
			# just calculate the strict lower triangle to save time, since the covariance matrix is symmetric
			dist2 = cdist2(x1, x1)
			mask = torch.tril(dist2 < r_c ** 2, -1)
		result: torch.Tensor = torch.zeros_like(dist2)
		result[mask] = self.__poly(torch.sqrt(dist2[mask]) / r_c) * torch.exp(-dist2[mask] / 2.0)
		if x2 is not None:
			return result
		else:
			return result + result.T + torch.eye(x1.shape[-2])
		# dist: torch.Tensor
		# mask: torch.Tensor
		# if x2 is not None:
		# 	assert x2.shape[-1] == lengthscale.numel() and x2.ndim >= 2
		# 	x2 = x2 / lengthscale
		# 	dist = _cdist(x1, x2)
		# 	mask = dist < r_c
		# else:
		# 	# just calculate the strict lower triangle to save time, since the covariance matrix is symmetric
		# 	dist = _cdist(x1, x1)
		# 	mask = torch.tril(dist < r_c, -1)
		# result: typing.Final[torch.Tensor] = torch.sparse_coo_tensor(mask.nonzero().T, self.__poly(dist[mask] / r_c) * torch.exp(-dist[mask] ** 2 / 2.0), dist.shape)
		# if x2 is not None:
		# 	return result.coalesce()
		# else:
		# 	return result + result.T + torch.sparse_coo_tensor(torch.arange(x1.shape[-2]).reshape(1, -1).repeat(2, 1), torch.ones(x1.shape[-2]), dist.shape)

	def integral(self, extra_dim: int, r_c: float) -> float:
		r"""The integral of :math: \int_0^{r_c} \phi_{d,k}(r/r_c) r^{\mathtt{extra\_dim}} dr

		Parameters
		----------
		extra_dim : int
			The extra dimension of the integral, which is the power of r
		r_c : float
			The support radius, which is the upper bound of the integral

		Returns
		-------
		float
			The value of the integral
		"""
		return float(typing.cast(float, self.__integrals[extra_dim].evalf(subs={wendland_rbf.__r_c: r_c})))

	def integral_d_1(self, r_c: torch.Tensor) -> torch.Tensor:
		r"""The integral of :math: \int_0^{r_c} \phi_{d,k}(r/r_c) r^{d-1} dr, but in numerical form

		Parameters
		----------
		r_c : torch.Tensor
			The support radius, which is the upper bound of the integral

		Returns
		-------
		torch.Tensor
			The value of the integral
		"""
		rc_2: typing.Final[torch.Tensor] = 1.0 / r_c ** 2
		plain: torch.Tensor = torch.tensor(0)
		erf: torch.Tensor = torch.tensor(0)
		decay: torch.Tensor = torch.tensor(0)
		for t in range(self.__even_ub, -1, -1):
			plain = plain * rc_2 + self.__a1[2 * t]
			if t > self.__poly.k:
				decay = decay * rc_2 + self.__j_coe[t]
		decay *= rc_2 ** (self.__poly.k + 1)
		for t in range(self.__odd_ub, self.__poly.k - 1, -1):
			erf = erf * rc_2 + self.__a1[2 * t + 1]
		erf *= rc_2 ** self.__poly.k / r_c
		return plain + wendland_rbf.__SQRT_PI_2 * torch.erf(r_c * wendland_rbf.__SQRT_1_2) * erf - torch.exp(-r_c**2/2) * decay

	def square_integral(self, x: torch.Tensor, /, *, lengthscale: torch.Tensor, r_c: torch.Tensor | float) -> torch.Tensor:
		r"""To evaluate the diagonal element of the integral of the square of the kernel function

		Parameters
		----------
		x : torch.Tensor, shape of (N, D)
			The feature set
		lengthscale : torch.Tensor, shape of (D,)
			The characteristic lengths
		r_c : torch.Tensor | float
			The support radius, which is the upper bound of the integral

		Returns
		-------
		torch.Tensor
			The value of the integral
		"""
		def rc_fit() -> torch.Tensor:
			r"""To extract the Chebyshev polynomial value at the support radius

			Returns
			-------
			torch.Tensor
				The Chebyshev polynomial value at the nodes for the support radius
			"""
			rc_dist: typing.Final[torch.Tensor] = r_c - self.__rc_values
			if torch.any(torch.abs(rc_dist) < wendland_rbf.__grid_close_eps):
				r_c_idx: typing.Final[int] = int(torch.argmin(torch.abs(rc_dist)).item())
				return self.__rc_chebyshev_func_vals[r_c_idx]
			else:
				div: typing.Final[torch.Tensor] = self.__rc_chebyshev_weights / rc_dist # shape (N_grid,)
				return (self.__rc_chebyshev_func_vals * div.reshape(-1, 1)).sum(0) / div.sum()

		assert x.shape[-1] == lengthscale.numel() and x.ndim == 2
		lengthscale = lengthscale.reshape(-1).detach()
		x = x / (r_c * lengthscale)
		dist2: typing.Final[torch.Tensor] = cdist2(x, x)
		mask: typing.Final[torch.Tensor] = torch.tril(dist2 < 2.0, -1) # diagonal has 0 dist, and can be calculated easily
		values: typing.Final[torch.Tensor] = torch.sqrt(dist2[mask])
		func_vals: typing.Final[torch.Tensor] = rc_fit()
		# check if close to grids
		fit: torch.Tensor = torch.empty(values.numel())
		grid_dist: typing.Final[torch.Tensor] = values.unsqueeze(-1) - self.__rc_chebyshev_grids # shape (N_test, N_grid)
		dist_close: typing.Final[torch.Tensor] = torch.abs(grid_dist) < wendland_rbf.__grid_close_eps
		close_pairs: typing.Final[torch.Tensor] = torch.nonzero(dist_close)
		fit[close_pairs[:, 0]] = func_vals[close_pairs[:, 1]]
		# chebyshev nodes weighted lagrange polynomial
		noclose: typing.Final[torch.Tensor] = torch.logical_not(dist_close.any(-1))
		div: typing.Final[torch.Tensor] = self.__rc_chebyshev_weights / grid_dist[noclose, :]
		fit[noclose] = (div * func_vals).sum(-1) / div.sum(-1)
		# only strict lower triangle
		result: torch.Tensor = torch.zeros_like(dist2)
		result[mask] = fit
		# coo: typing.Final[torch.Tensor] = torch.sparse_coo_tensor(mask.nonzero().T, fit, dist.shape)
		# # add diagonal fit from analytical integral
		# return coo + coo.T + torch.sparse_coo_tensor(torch.arange(x.shape[-2]).reshape(1, -1).repeat(2, 1), torch.full((x.shape[-2],), func_vals[-1].item()), dist.shape)
		return result + result.T + torch.eye(x.shape[-2]) * func_vals[-1]
