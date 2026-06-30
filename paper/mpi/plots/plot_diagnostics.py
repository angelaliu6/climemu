"""Diagnostic plots for the consistency model: c_skip/c_out, bin schedule, noise grids."""
import numpy as np
import matplotlib.pyplot as plt
import os, sys

base_dir = os.getcwd()
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config

config = Config()
SIGMA_MIN = config.schedule.time_min   # 0.002  (σ_min; time = σ in VE schedule)
DATA_STD  = config.schedule.data_std   # 0.5
BINS_MIN  = config.training.bins_min   # 2
BINS_MAX  = config.training.bins_max   # 150
BINS_RHO  = config.training.bins_rho   # 7.0
SIGMA_MAX = float(np.load(config.data.sigma_max_path))


# =============================================================================
# HELPERS
# =============================================================================

def c_skip_out(σ, σ_min=SIGMA_MIN, data_std=DATA_STD):
    dσ    = σ - σ_min
    denom = dσ**2 + data_std**2
    return data_std**2 / denom, data_std * dσ / np.sqrt(denom)


def compute_bins(frac, bins_min=BINS_MIN, bins_max=BINS_MAX):
    return np.ceil(np.sqrt(frac * (bins_max**2 - bins_min**2) + bins_min**2)).astype(int)


def frac_for_bins(b, bins_min=BINS_MIN, bins_max=BINS_MAX):
    """Approximate training fraction at which bins first reaches b."""
    return (b**2 - bins_min**2) / (bins_max**2 - bins_min**2)


def karras_sigmas(bins, σ_min=SIGMA_MIN, σ_max=SIGMA_MAX, rho=BINS_RHO):
    n = np.arange(bins)
    return (σ_min**(1/rho) + n / (bins - 1) * (σ_max**(1/rho) - σ_min**(1/rho)))**rho


# =============================================================================
# FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- Panel 1: c_skip and c_out vs σ ---
ax = axes[0]
σ_vals = np.logspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), 500)
cs, co = c_skip_out(σ_vals)
ax.semilogx(σ_vals, cs, label='$c_{skip}$', color='steelblue')
ax.semilogx(σ_vals, co, label='$c_{out}$',  color='tomato')
ax.axvline(SIGMA_MIN, color='gray', lw=0.8, ls='--', label=f'$\\sigma_{{min}}={SIGMA_MIN}$')
ax.axvline(SIGMA_MAX, color='gray', lw=0.8, ls=':',  label=f'$\\sigma_{{max}}={SIGMA_MAX:.1f}$')
ax.set_xlabel('$\\sigma$')
ax.set_ylabel('Coefficient value')
ax.set_title('EDM preconditioning')
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# --- Panel 2: bins vs training fraction ---
ax = axes[1]
frac = np.linspace(0, 1, 500)
bins_curve = compute_bins(frac)
ax.plot(frac, bins_curve, color='mediumpurple')
ax.set_xlabel('Training progress (step / total_steps)')
ax.set_ylabel('bins')
ax.set_title(f'Bin schedule ({BINS_MIN} → {BINS_MAX})')
ax.grid(True, alpha=0.3)

# --- Panel 3: noise level sequence ---
# Bins values chosen to give clearly different point densities
ax = axes[2]
bins_to_plot = [2, 10, 40, 150]
palette      = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
marker_sizes = [80, 40, 15, 5]

for color, b, ms in zip(palette, bins_to_plot, marker_sizes):
    σs  = karras_sigmas(b)
    x   = np.arange(b) / b
    pct = frac_for_bins(b) * 100
    ax.scatter(x, σs, s=ms, color=color, label=f'bins={b}  ({pct:.0f}% training)',
               zorder=3, alpha=0.8)

ax.set_yscale('log')
ax.set_xlabel('Bin index / total bins')
ax.set_ylabel('$\\sigma$  (log scale)')
ax.set_title('Noise Level Sequence')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out = 'paper/mpi/plots/diagnostics.jpg'
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved {out}")
plt.show()
