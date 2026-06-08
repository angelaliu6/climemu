import jax.numpy as jnp

CACHE_DIR = "paper/mpi/cache"
_stats = jnp.load(f"{CACHE_DIR}/μ_σ.npz")
σ_train = _stats["σ"]
σ2 = σ_train[:-1]**2
σdata = jnp.sqrt(σ2.mean()) # 2.3076885