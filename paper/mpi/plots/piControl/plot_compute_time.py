import os
import sys
import time
import numpy as np
import einops
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from functools import partial

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.diffusion import HealPIXUNet, ContinuousVESchedule
from paper.mpi.config import Config
from paper.mpi.main import Denoiser
from paper.mpi import utils
from paper.mpi.plots.piControl.utils import save_plot


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/piControl/files'
DPI = 300
N_RUNS = 30
CACHE_DIR = "paper/mpi/cache"


# =============================================================================
# MODEL LOADING
# =============================================================================

config = Config()

_edges = jnp.load(os.path.join(CACHE_DIR, "edges.npz"))
edges_to_healpix = _edges["to_healpix"]
edges_to_latlon = _edges["to_latlon"]

def _build_unet():
    return HealPIXUNet(
        input_size=config.model.input_size,
        nside=config.model.nside,
        enc_filters=list(config.model.enc_filters),
        dec_filters=list(config.model.dec_filters),
        out_channels=config.model.out_channels,
        temb_dim=config.model.temb_dim,
        healpix_emb_dim=config.model.healpix_emb_dim,
        edges_to_healpix=edges_to_healpix,
        edges_to_latlon=edges_to_latlon,
    )

# Diffusion model: raw HealPIXUNet
diffusion_model = _build_unet()
diffusion_model = eqx.tree_deserialise_leaves(config.training.model_filename, diffusion_model)

# Consistency model: Denoiser wrapper
consistency_base = _build_unet()
consistency_model = Denoiser(
    consistency_base,
    config.model.context_channels,
    time_min=config.schedule.time_min,
    data_std=config.schedule.data_std,
)
consistency_model = eqx.tree_deserialise_leaves(
    config.training.consistency_model_filename, consistency_model
)

σmax = float(jnp.load(os.path.join(CACHE_DIR, "σmax.npy")))
schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)

_stats = jnp.load(os.path.join(CACHE_DIR, "μ_σ.npz"))
μ_train = _stats["μ"]
σ_train = _stats["σ"]

β = jnp.load(os.path.join(CACHE_DIR, "β.npy"))
β_4d = einops.rearrange(β, 'm (l1 l2) i -> m l1 l2 i', l1=96)

ΔT = jnp.array([2.0])
month = jnp.array([5])
pattern_batch = β_4d[month, :, :, 0] + β_4d[month, :, :, 1] * ΔT.reshape(-1, 1, 1)

output_size = (config.model.out_channels, config.model.input_size[1], config.model.input_size[2])
χ = jr.PRNGKey(config.sampling.random_seed)


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

generate_diffusion = partial(
    utils.draw_samples_batch,
    model=diffusion_model,
    schedule=schedule,
    pattern_batch=pattern_batch,
    n_samples=1,
    n_steps=config.sampling.n_steps,
    μ=μ_train, σ=σ_train,
    output_size=output_size,
    key=χ,
)

generate_consistency = partial(
    utils.draw_samples_batch_consistency,
    denoiser=consistency_model,
    schedule=schedule,
    pattern_batch=pattern_batch,
    n_samples=1,
    n_steps=1,
    μ=μ_train, σ=σ_train,
    output_size=output_size,
    time_min=config.schedule.time_min,
    time_max=float(σmax),
    bins_rho=config.training.bins_rho,
    key=χ,
)


# =============================================================================
# WARMUP + TIMING
# =============================================================================

print("Warming up diffusion model...")
out = generate_diffusion()
jax.block_until_ready(out)

print("Warming up consistency model...")
out = generate_consistency()
jax.block_until_ready(out)

print(f"Timing diffusion model ({N_RUNS} runs)...")
diffusion_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    out = generate_diffusion()
    jax.block_until_ready(out)
    diffusion_times.append(time.perf_counter() - t0)

print(f"Timing consistency model ({N_RUNS} runs)...")
consistency_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    out = generate_consistency()
    jax.block_until_ready(out)
    consistency_times.append(time.perf_counter() - t0)

diffusion_times = np.array(diffusion_times)
consistency_times = np.array(consistency_times)

print(f"Diffusion:   {diffusion_times.mean():.3f}s ± {diffusion_times.std():.3f}s per sample")
print(f"Consistency: {consistency_times.mean():.3f}s ± {consistency_times.std():.3f}s per sample")


# =============================================================================
# PLOTTING
# =============================================================================

def create_compute_time_plot():
    fig, ax = plt.subplots(figsize=(4, 4))

    labels = ["Diffusion", "Consistency"]
    means = [diffusion_times.mean(), consistency_times.mean()]
    stds = [diffusion_times.std(), consistency_times.std()]
    colors = ["darkorange", "tomato"]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
                  error_kw=dict(elinewidth=1.5, ecolor='black'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Time per sample [s]", fontsize=12)
    ax.set_title("Sampling compute time\n(after JIT compilation)", fontsize=12)
    ax.margins(x=0.3, y=0.1)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.01 * max(means),
                f"{mean:.3f}s", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def main():
    fig = create_compute_time_plot()
    save_plot(fig, OUTPUT_DIR, 'compute_time.jpg', dpi=DPI)


if __name__ == "__main__":
    main()
