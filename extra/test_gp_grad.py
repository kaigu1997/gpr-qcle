import torch

torch.set_default_dtype(torch.double)


def main() -> None:
	M = 100
	N = 10
	D = 6
	x_test = torch.rand((M, D)).detach().requires_grad_()
	x_train = torch.rand((N, D))
	length = torch.rand(D)
	cov = torch.exp(-((x_test[:, torch.newaxis, :] - x_train) / length).square().sum(-1) / 2.0)
	print(torch.dist(torch.autograd.grad(cov, x_test, torch.eye(N).reshape(N, 1, N).repeat(1, M, 1), True, True, is_grads_batched=True)[0].moveaxis(0, -1), (cov[..., None] * (x_train - x_test[:, None, :]) / length ** 2).moveaxis(-2, -1)))

	cov2 = torch.exp(-((x_test[:, torch.newaxis, :] - x_test[:N]) / length).square().sum(-1) / 2.0)
	x_test_n = x_test[:N].detach().requires_grad_()
	cov3 = torch.exp(-((x_test[:, torch.newaxis, :] - x_test_n) / length).square().sum(-1) / 2.0)
	print(torch.dist(torch.autograd.grad(cov2.reshape(-1), x_test, torch.eye(cov2.numel()), True, True, is_grads_batched=True)[0][:, :N], torch.autograd.grad(cov3.reshape(-1), x_test, torch.eye(cov2.numel()), True, True, is_grads_batched=True)[0][:, :N] + torch.autograd.grad(cov3.reshape(-1), x_test_n, torch.eye(cov2.numel()), True, True, is_grads_batched=True)[0]))
	print(torch.dist(torch.autograd.grad(cov2.reshape(-1), x_test, torch.eye(cov2.numel()), True, True, is_grads_batched=True)[0][:, N:], torch.autograd.grad(cov3.reshape(-1), x_test, torch.eye(cov2.numel()), True, True, is_grads_batched=True)[0][:, N:]))


if __name__ == "__main__":
	main()