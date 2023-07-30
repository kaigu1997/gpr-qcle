import evolve

import copy
import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import numpy.typing as npt
import torch
import typing

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class GP(gpytorch.models.ExactGP):
	def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood, kernel: gpytorch.kernels.Kernel):
		super().__init__(x, y, likelihood)
		self.mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
		self.cov: gpytorch.kernels.Kernel = kernel

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		return gpytorch.distributions.MultivariateNormal(self.mean(x), self.cov(x))


def sr_pred(
	model: GP,
	likelihood: gpytorch.likelihoods.GaussianLikelihood,
	x: torch.Tensor,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
	x_test: torch.Tensor
) -> torch.Tensor:
	return (model.cov(x_test, x) @ (linear_operator.utils.stable_pinverse(model.cov(x_all, x).to_dense()) @ y_all)).to_dense()


def sr_se(
	x: torch.Tensor,
	y: torch.Tensor,
	model: GP,
	likelihood: gpytorch.likelihoods.GaussianLikelihood,
	x_all: torch.Tensor,
	y_all: torch.Tensor
) -> torch.Tensor:
	return ((y_all - sr_pred(model, likelihood, x, x_all, y_all, x_all)) ** 2).sum()


class Predictor:
	MAX_ITER: typing.Literal[50000] = 50000
	FTOL: float = 2.2204460492503131e-09
	GTOL: float = 1e-5
	NOISE: float = 1e-4

	def __init__(self, kernel: gpytorch.kernels.Kernel, loss_func: typing.Callable = sr_se, predictor: typing.Callable | None = sr_pred):
		self.kernel: gpytorch.kernels.Kernel = kernel
		self.loss_func: typing.Callable = loss_func
		self.predictor: typing.Callable | None = predictor
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((2,), Predictor.NOISE))
		self.x_all: torch.Tensor = torch.Tensor()
		self.y_all: torch.Tensor = torch.Tensor()
		self.model: GP = GP(torch.zeros((2, evolve.PHASEDIM)), torch.zeros((2,)), likelihood, self.kernel)
		self.model_param: dict[str, torch.Tensor] = copy.deepcopy(self.model.state_dict())

	def train(self) -> None:
		def print_model(model: gpytorch.models.ExactGP, print_grad: bool = False) -> None:
			def print_tensor(t: torch.Tensor):
				if t.dim() == 0 or np.prod(t.size()) == 1:
					return t.item()
				else:
					return t.detach().numpy().ravel()

			fmt: str = 'Parameter name: {0:42} value = {1}'
			fmt_grad: str = fmt + ' grad = {2}'
			for param_name, param, constraint in model.named_parameters_and_constraints():
				if print_grad and param.grad is not None:
					print(fmt_grad.format(param_name, print_tensor(param), print_tensor(param.grad)))
				else:
					print(fmt.format(''.join(param_name.split('raw_')), print_tensor(constraint.transform(param) if isinstance(constraint, gpytorch.constraints.Interval) else param)))

		def get_lr(optimizer: torch.optim.Optimizer) -> float:
			return optimizer.param_groups[0]['lr']

		def print_stuff(loss: torch.Tensor, optimizer: torch.optim.Optimizer, model: gpytorch.models.ExactGP, print_grad: bool = False, extra_str="\t") -> None:
			print(extra_str + "loss = {:.15e}, lr = {}".format(loss.item(), get_lr(optimizer)))
			print_model(model, print_grad)
			print('\n')

		assert self.model.train_inputs is not None and isinstance(self.model.train_targets, torch.Tensor)
		x: torch.Tensor = self.model.train_inputs[0]
		y: torch.Tensor = self.model.train_targets
		if not torch.all(y == 0):
			# train model
			likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((x.shape[0],), Predictor.NOISE))
			self.model.likelihood = likelihood
			self.model.train()
			likelihood.train()
			self.model.load_state_dict(self.model_param)
			# print_model(model)
			finish_early: bool = False
			optimizer: torch.optim.Optimizer = torch.optim.Rprop(self.model.parameters(), lr=1.0)
			optimizer.zero_grad()
			loss: torch.Tensor = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
			loss.backward()
			last_value: float = loss.item()
			print_stuff(loss, optimizer, self.model, True, "\n\nInit")
			for i in range(1, Predictor.MAX_ITER + 1):
				# adjust lr
				old_prm: dict[str, torch.Tensor] = copy.deepcopy(self.model.state_dict())
				optimizer.step()
				loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
				# print_stuff(loss, optimizer, self.model, True)
				if loss < last_value:
					# print('loss < last_value\n')
					while loss < last_value:
						second_last_value: float = last_value
						last_value = loss.item()
						self.model.load_state_dict(old_prm)
						optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) * 2.0)
						optimizer.step()
						loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
						# print_stuff(loss, optimizer, self.model, True)
						if loss >= last_value:
							# goes back, not only loss but also last value, for stop criteria judgment
							last_value = second_last_value
							break
					# when exit, loss >= last value, so learning rate should be halved
					self.model.load_state_dict(old_prm)
					optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) / 2.0)
					optimizer.step()
					loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
					# print_stuff(loss, optimizer, self.model, True)
				else:
					# print('loss > last_value')
					while loss >= last_value:
						last_loop_value: float = loss.item()
						self.model.load_state_dict(old_prm)
						optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer) / 2.0)
						optimizer.step()
						loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
						# print_stuff(loss, optimizer, self.model, True)
						if last_loop_value == loss.item():
							# no stepping forward, but still larger than last, meaning last is the best
							self.model.load_state_dict(old_prm)
							loss = self.loss_func(x, y, self.model, likelihood, self.x_all, self.y_all)
							break
				if i % (Predictor.MAX_ITER // 100) == 0:
					print('\n')
					print('Iter {} - Loss: {:.15e} - lr: {}'.format(i, loss.item(), get_lr(optimizer)))
					print_model(self.model, True)
					print_model(self.model)
					print('\n')
				# stopping criteria
				if (last_value - loss.item()) / max(abs(last_value), abs(loss.item()), 1.0) < Predictor.FTOL:
					finish_early = True
					print("Convergence: |f_i - f_{i+1}| <= FTOL")
					print('Iter {} - Loss: {:.15e} - lr: {}'.format(i, loss.item(), get_lr(optimizer)))
					break
				if np.sqrt(sum([torch.sum(param.grad ** 2).item() if param.grad is not None else 0.0 for param in self.model.parameters()])) < Predictor.GTOL:
					finish_early = True
					print("Convergence: |Gradient| <= GTOL")
					print('Iter {} - Loss: {:.15e} - lr: {}'.format(i, loss.item(), get_lr(optimizer)))
					break
				optimizer = optimizer.__class__(self.model.parameters(), lr=get_lr(optimizer))
				optimizer.zero_grad()
				last_value = loss.item()
				loss.backward()
				# print_stuff(loss, optimizer, self.model, True, "\n\n\tlast = {}, ".format(last_value))
			if not finish_early:
				print('Iter {} - Loss: {:.15e} - lr: {}'.format(Predictor.MAX_ITER, last_value, [param['lr'] for param in optimizer.param_groups]))
				print("Stop: Total No. iterations reached limit.")
			print_model(self.model)
			print('\n')
			self.model_param = copy.deepcopy(self.model.state_dict())


predictors: list[Predictor] = [Predictor(gpytorch.kernels.RBFKernel(evolve.PHASEDIM), sr_se, sr_pred) for i in range(evolve.NUM_ELM)]


def update(x_all: npt.NDArray[np.double], y_all: npt.NDArray[np.double], num_pts: int) -> None:
	for x, y, pred in zip(x_all, y_all, predictors):
		pred.model.set_train_data(torch.from_numpy(x[:num_pts]), torch.from_numpy(y[:num_pts]), False)
		pred.x_all = torch.from_numpy(x)
		pred.y_all = torch.from_numpy(y)


def train() -> None:
	for pred in predictors:
		if isinstance(pred.model.train_targets, torch.Tensor) and not torch.all(pred.model.train_targets == 0):
			pred.train()


def predict(x_input: npt.NDArray[np.double], ElementIndex: int) -> npt.NDArray[np.double]:
	x_test: torch.Tensor = torch.from_numpy(x_input)
	pred: Predictor = predictors[ElementIndex]
	if not isinstance(pred.model.train_targets, torch.Tensor) or (isinstance(pred.model.train_targets, torch.Tensor) and torch.all(pred.model.train_targets == 0)):
		return np.zeros(x_input.shape[0])
	else:
		# train model
		assert pred.model.train_inputs is not None and isinstance(pred.model.train_inputs[0], torch.Tensor)
		likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full((pred.model.train_inputs[0].shape[0],), Predictor.NOISE))
		# predict
		pred.model.eval()
		likelihood.eval()
		if pred.predictor is None:
			return likelihood(pred.model.__call__(x_test), noise=torch.full((x_test.shape[0],), Predictor.NOISE)).mean.detach().numpy()
		else:
			return pred.predictor(pred.model, likelihood, pred.model.train_inputs[0], pred.x_all, pred.y_all, x_test).detach().numpy()
