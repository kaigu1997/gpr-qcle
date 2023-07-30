import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
DIM = 3


def f(x: jax.Array) -> jax.Array:
	return jnp.moveaxis(jnp.array([[jnp.sin(x[:, 0]), jnp.cos(x[:, 0])], [jnp.exp(x[:, 1]), jnp.exp(-x[:, 1])], [jnp.exp(x[:, 1]), jnp.exp(-x[:, 1])], [jnp.sin(x[:, 0]), jnp.cos(x[:, 0])]]), (0, 1), (-2, -1))


def printf(x: jax.Array) -> None:
	x = x.reshape(-1, DIM)
	y = f(x)
	j = jnp.moveaxis(jax.vmap(jax.jacrev(f), 0)(x.reshape(-1, 1, DIM)), -1, -3)
	print(x.shape, x, y.shape, y, j.shape, j, sep='\n', end='\n\n')


printf(jnp.array([0.0, jnp.pi, jnp.pi * 2]))
printf(jnp.repeat(jnp.linspace(0.0, jnp.pi, 5)[:, jnp.newaxis], DIM, -1))
