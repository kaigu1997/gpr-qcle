import collections.abc

import gpytorch
import gpytorch.constraints
import numpy as np
import numpy.typing as npt
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import scipy.spatial
import sklearn.cluster
import torch

import gp
import opt
import se


def linear_trial(rng: np.random.Generator) -> None:
	x: npt.NDArray[np.double] = rng.uniform(-1.0, 1.0, 100)
	x_km: npt.NDArray[np.double] = sklearn.cluster.KMeans(20, init="k-means++", n_init="auto", random_state=np.random.RandomState(rng.bit_generator), algorithm="lloyd").fit(x.reshape(-1, 1)).cluster_centers_
	print("Finish KMeans")
	x.sort()
	x_t: torch.Tensor = torch.from_numpy(x_km).detach()
	dist: float = scipy.spatial.distance.pdist(x_km).min()
	print(f"min{{dist}} = {dist}")
	for f_s in ["lambda x: 0", "lambda x: x", "lambda x: 100 * x"]:
		print(f"\t{f_s}")
		f: collections.abc.Callable[[npt.NDArray[np.double]], npt.NDArray[np.double]] = eval(f_s)
		y_t: torch.Tensor = torch.from_numpy(f(x_km)).detach()
		y: npt.NDArray[np.double] = f(x)
		for i in np.arange(1, 11):
			print(f"\t\tlength < {i} * min{{dist}} = {i * dist}")
			model: gp.GPR | None = opt.gpytorch_train(x_t, y_t, gpytorch.kernels.RBFKernel(1, lengthscale_constraint=gpytorch.constraints.Interval(0.0, i * dist)), se.loocv_generator(), False)
			assert model is not None
			cov_mat: torch.Tensor = model.cov(x_t).to_dense()
			inv: torch.Tensor = gp.square_solver(cov_mat, torch.eye(20))
			print(f"\t\t||cov @ inv - I|| = {torch.dist(cov_mat @ inv, torch.eye(20))}, ||inv @ cov - I|| = {torch.dist(inv @ cov_mat, torch.eye(20))}")
			pred: npt.NDArray[np.double] = gp.default_predict(model.cov, x_t, y_t, torch.from_numpy(x.reshape(-1, 1))).detach().numpy().reshape(-1)
			print(f"\t\t||pred - y|| = {np.linalg.norm(pred - y)}")
			fig: matplotlib.figure.Figure = plt.figure(figsize=(6.4, 4.8))
			ax: matplotlib.axes.Axes = fig.subplots()
			ax.plot(x.reshape(-1), y.reshape(-1), label="y")
			ax.plot(x, pred, label="u")
			ax.scatter(x_km.reshape(-1), f(x_km.reshape(-1)))
			ax.legend()
			fig.savefig(f"linear_{f_s.replace(" ", "_")}_{i}.png")
