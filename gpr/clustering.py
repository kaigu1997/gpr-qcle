r"""clustering
==========
Author: Josue N Rivera (github.com/wzjoriv)
Date: 7/3/2021
Description: Snippet of various clustering implementations only using PyTorch
Full project repository: https://github.com/wzjoriv/Lign (A graph deep learning framework that works alongside PyTorch)
"""
import dataclasses
import typing

import torch

import constant

torch.set_default_dtype(constant.DTYPE)
torch.set_default_device(constant.DEVICE)


def distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	r"""To compute the pairwise distance matrix between two sets of vectors

	Parameters
	----------
	x : torch.Tensor
		The first set of vectors
	y : torch.Tensor
		The second set of vectors

	Returns
	-------
	torch.Tensor
		The pairwise distance matrix
	"""
	return torch.linalg.norm(x[:, torch.newaxis] - y[torch.newaxis], dim=-1)


@dataclasses.dataclass
class _KNNResult:
	indices: torch.Tensor
	distances: torch.Tensor
	points: torch.Tensor


def knn(train_points: torch.Tensor, k: int, test_points: torch.Tensor | None = None) -> _KNNResult:
	r"""To find the k-nearest neighbors of a set of points

	Parameters
	----------
	train_points : torch.Tensor
		The training points
	k : int
		The number of neighbors to consider
	test_points : torch.Tensor | None, optional
		The points to predict the labels for, by default None (and uses the training points)

	Returns
	-------
	_KNNResult
		The k-nearest neighbors result, including indices, distances, and points
	"""
	assert train_points.ndim == 2
	if test_points is not None:
		assert test_points.ndim == 2
	else: # test_points is None
		test_points = train_points
	topk: typing.Final = distance_matrix(test_points, train_points).topk(k, -1, False, False)
	return _KNNResult(indices=topk.indices, distances=topk.values, points=train_points[topk.indices])


# class KMeans:
# 	@staticmethod
# 	def random_sample(tensor: torch.Tensor, k: int) -> torch.Tensor:
# 		r"""To randomly sample k elements from a tensor

# 		Parameters
# 		----------
# 		tensor : torch.Tensor
# 			The input tensor to sample from
# 		k : int
# 			The number of elements to sample

# 		Returns
# 		-------
# 		torch.Tensor
# 			The randomly sampled k elements
# 		"""
# 		return tensor[torch.randperm(tensor.shape[0])[:k]]

# 	def __init__(self, X = None, k=2, n_iters = 10):

# 		self.k = k
# 		self.n_iters = n_iters

# 		if type(X) != type(None):
# 			self.train(X)

# 	def train(self, X):

# 		self.train_pts = KMeans.random_sample(X, self.k)
# 		self.train_label = torch.LongTensor(range(self.k))

# 		for _ in range(self.n_iters):
# 			labels = self.train_label[torch.argmin(distance_matrix(X, self.train_pts), dim=1)]

# 			for lab in range(self.k):
# 				select = labels == lab
# 				self.train_pts[lab] = torch.mean(X[select], dim=0)
