import math
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from functools import partial


def compute_bins(step, total_steps, bins_min=2, bins_max=150):
    """Progressive time discretization: bins grows from bins_min to bins_max.

    Matches Philip's formula: ceil(sqrt(frac * (N²-n²) + n²))
    """
    frac = jnp.minimum(step / jnp.maximum(total_steps, 1), 1.0)
    return jnp.ceil(
        jnp.sqrt(frac * (bins_max**2 - bins_min**2) + bins_min**2)
    ).astype(jnp.int32)


def timesteps_to_times(n, bins, time_min, time_max, rho=7.0):
    """Karras schedule: maps integer bin index n → continuous noise time.

    Matches Philip's timesteps_to_times().
    """
    return (
        time_min ** (1.0 / rho)
        + n / (bins - 1) * (time_max ** (1.0 / rho) - time_min ** (1.0 / rho))
    ) ** rho


def compute_ema_decay(step, total_steps, bins_min=2, initial_decay=0.9):
    """Dynamic EMA decay tied to current bin count.

    Matches Philip's formula: exp(bins_min * log(mu_0) / bins)
    Starts near initial_decay when bins=bins_min, and decreases as bins grows.
    """
    bins = compute_bins(step, total_steps, bins_min)
    return jnp.exp(bins_min * jnp.log(initial_decay) / bins)


def ct_single_loss(model, ema_model, ctx_size, x, step, total_steps,
                   time_min, time_max, bins_min, bins_max, rho, key):
    """Single-sample CT loss (Philip's adjacent-timestep formulation).

    Trains model(x_{n+1}, t_{n+1}) ≈ EMA_model(x_n, t_n) for adjacent
    Karras timestep pairs, with progressive binning during training.
    """
    x0, ctx = x[:-ctx_size, ...], x[-ctx_size:, ...]

    bins = compute_bins(step, total_steps, bins_min, bins_max)
    key1, key2 = jr.split(key)

    # Sample adjacent pair index n ~ U[0, bins-2]
    n = jr.randint(key1, (), 0, bins - 1)
    t_n  = timesteps_to_times(n,     bins, time_min, time_max, rho)
    t_n1 = timesteps_to_times(n + 1, bins, time_min, time_max, rho)

    # Same noise ε for both (key consistency property)
    eps = jr.normal(key2, x0.shape)
    x_current = jnp.concatenate([x0 + eps * t_n,  ctx], axis=0)
    x_next    = jnp.concatenate([x0 + eps * t_n1, ctx], axis=0)

    # EMA target at lower noise (stop_gradient breaks the cycle)
    target = jax.lax.stop_gradient(ema_model(x_current, t_n))
    pred   = model(x_next, t_n1)

    c = 1e-3
    mse = jnp.mean((pred - target) ** 2)
    loss = jnp.sqrt(mse + c**2) - c  # pseudo-Huber
    return loss, mse


@eqx.filter_jit
def ct_batch_loss(model, ema_model, ctx_size, x, step, total_steps,
                  time_min, time_max, bins_min, bins_max, rho, key):
    """Vectorised batch CT loss."""
    batch_size = x.shape[0]
    key1, key2 = jr.split(key)

    L = jax.vmap(
        partial(ct_single_loss, model, ema_model),
        in_axes=(None, 0, None, None, None, None, None, None, None, 0)
    )

    keys = jr.split(key2, batch_size)
    batch_loss, batch_mse = L(
        ctx_size, x, step, total_steps, time_min, time_max, bins_min, bins_max, rho, keys
    )
    return batch_loss.mean(), batch_mse.mean()


@eqx.filter_jit
def ct_make_step(model, ema_model, ctx_size, x, step, total_steps,
                 time_min, time_max, bins_min, bins_max, rho,
                 key, opt_state, opt_update):
    """Single optimisation step for CT training.

    Returns (loss, mse, updated_model, key, opt_state, grad_norm).
    """
    loss_fn = eqx.filter_value_and_grad(ct_batch_loss, has_aux=True)
    (loss, mse), grads = loss_fn(
        model, ema_model, ctx_size, x, step, total_steps,
        time_min, time_max, bins_min, bins_max, rho, key
    )

    grad_norm = _compute_grad_norm(grads)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key, _ = jr.split(key)
    return loss, mse, model, key, opt_state, grad_norm


@eqx.filter_jit
def _compute_grad_norm(grads):
    squared = [jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads) if g is not None]
    return jnp.sqrt(jnp.sum(jnp.array(squared)))
