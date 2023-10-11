#!/usr/bin/env python3
"""
Quasi Regular
=============
To construct the quasi regular sample points
"""
import typing

import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)
DIM: typing.Literal[2] = 2


def draw(x: torch.Tensor, title: str) -> None:
	"""
	_summary_

	Parameters
	----------
	x : torch.Tensor
		_description_
	title : str
		_description_
	"""
	fig, ax = plt.subplots()
	ax.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy())
	fig.savefig(title + '.png')
	plt.close(fig)


def a_func(x: torch.Tensor) -> torch.Tensor:
	"""
	_summary_

	Parameters
	----------
	x : torch.Tensor
		_description_

	Returns
	-------
	torch.Tensor
		_description_
	"""
	a_func.N_GAUSS = 50
	a_func.CENTER = torch.rand((a_func.N_GAUSS, DIM)) * 4.0 - 2.0
	a_func.WIDTH = torch.rand((a_func.N_GAUSS, DIM)) * 1.9 + 0.1
	a_func.WEIGHT = torch.rand(a_func.N_GAUSS) * 2.0 - 1.0
	if x.ndim == 1:
		x = x.unsqueeze(0)
	return torch.abs((a_func.WEIGHT * torch.exp(-(((x.unsqueeze(1) - a_func.CENTER) / a_func.WIDTH) ** 2).sum(-1) / 2.0)).sum(-1))


def normal_dist(
	X: torch.Tensor
) -> torch.Tensor:
	"""
	_summary_

	Parameters
	----------
	X : torch.Tensor
		_description_

	Returns
	-------
	torch.Tensor
		_description_
	"""
	return torch.exp(-(X ** 2).sum(-1) / 2.0)


def pseudo_potential(
	X: torch.Tensor,
	dist_func: typing.Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
	"""
	_summary_

	Parameters
	----------
	X : torch.Tensor
		_description_
	dist : typing.Callable[[torch.Tensor], torch.Tensor]
		_description_

	Returns
	-------
	torch.Tensor
		_description_
	"""
	pseudo_potential.m = DIM
	sigma: torch.Tensor = torch.pow(dist_func(X), -pseudo_potential.m / DIM) # shape of (N,)
	pdist_sq: torch.Tensor = ((X.unsqueeze(1) - X) ** 2).sum(-1).fill_diagonal_(1.0) # shape of (N, N)
	result: torch.Tensor = (sigma + sigma.unsqueeze(1) ** 2) / torch.pow(pdist_sq, pseudo_potential.m / 2)
	return torch.triu(result, 1).sum()


def simulate_tempering(
	x: torch.Tensor,
	potential: typing.Callable[[torch.Tensor], torch.Tensor],
	high_temp: float = 300.0,
	low_temp: float = 10.0,
	alpha: float = 0.95,
) -> torch.Tensor:
	"""
	_summary_

	Parameters
	----------
	x : torch.Tensor
		_description_
	potential : _type_, optional
		_description_, by default pseudo_potential
	high_temp : float, optional
		_description_, by default 300.0
	low_temp : float, optional
		_description_, by default 10.0
	alpha : float, optional
		_description_, by default 0.99
	"""
	Vx: torch.Tensor = potential(x)
	current_temp: float = high_temp
	disp: torch.Tensor = torch.Tensor([torch.min(torch.pdist(x[:, i:i+1], torch.inf)) for i in range(x.shape[-1])])
	iter_per_temp: int = int(torch.max(1.0 / disp).item())
	while current_temp > low_temp:
		print("T = {}, V = {}".format(current_temp, Vx.item()))
		for _ in range(iter_per_temp):
			x_new: torch.Tensor = x + torch.randn(x.shape) * disp
			Vx_new: torch.Tensor = potential(x_new)
			acc: torch.Tensor = torch.exp((Vx - Vx_new) / current_temp) > torch.rand(1)
			if acc.item():
				x = x_new
				Vx = Vx_new
		draw(x, str(int(current_temp)))
		current_temp *= alpha
	return x


def main() -> None:
	"""
	_summary_
	"""
	main.dist_func = normal_dist
	x: torch.Tensor = torch.randn((64, DIM), dtype=torch.float)
	# simulate tempering
	x = simulate_tempering(x, lambda x: pseudo_potential(x, main.dist_func), low_temp=0.1) * torch.Tensor([0.5, 1.0]) + torch.Tensor([-8.0, 20.0])
	draw(x, 'quasi_regular')


if __name__ == '__main__':
	main()
