import os
import sys
import numpy as np
import xarray as xr
import jax.numpy as jnp
import jax.random as jr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.data import load_dataset
from paper.mpi.plots.piControl.utils import load_data, VARIABLES, save_plot


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/piControl/files'
DIFFUSION_OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_diffusion"
DPI = 200
N_SAMPLES = 16
SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
climatology, piControl_consistency, piControl_cmip6 = load_data(config, in_memory=False)

piControl_diffusion = xr.open_dataset(os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_piControl.nc"))

# Subtract climatology from CMIP6 to get anomalies
piControl_cmip6_anom = piControl_cmip6 - climatology


# =============================================================================
# REPRODUCE ΔT SEQUENCE
# =============================================================================

β = jnp.load(config.data.pattern_scaling_path)
piControl_dataset = load_dataset(
    root=config.data.root_dir,
    model=config.data.model_name,
    experiments=["piControl"],
    variables=["tas"],
    in_memory=False,
    external_β=np.array(β)
)
σpiControl = float(piControl_dataset.gmst['piControl']['tas'].std().item())

n_year = piControl_consistency.sizes['year']
key = jr.PRNGKey(0)
ΔTs = []
for _ in range(n_year):
    key, _ = jr.split(key)
    ΔT_year = σpiControl * jr.normal(key, (12,))
    ΔTs.append(np.array(ΔT_year))
ΔTs = np.stack(ΔTs)  # shape (n_year, 12)

# Precompute area-weighted CMIP6 ΔT per (month, year)
weights = np.cos(np.deg2rad(piControl_cmip6_anom.lat))
cmip6_ΔT = (
    piControl_cmip6_anom['tas']
    .weighted(weights)
    .mean(['lat', 'lon'])
    .compute()
)  # dims: (month, year)


def find_cmip6_year(month_1indexed, target_ΔT):
    """Find the CMIP6 piControl year index whose area-weighted mean ΔT is closest to target."""
    monthly_ΔT = cmip6_ΔT.sel(month=month_1indexed).values
    return int(np.argmin(np.abs(monthly_ΔT - target_ΔT)))


# =============================================================================
# SELECT 16 SAMPLES
# =============================================================================

rng = np.random.default_rng(SEED)
year_indices = rng.integers(0, n_year, size=N_SAMPLES)
month_indices_0 = rng.integers(0, 12, size=N_SAMPLES)   # 0-indexed
month_indices_1 = month_indices_0 + 1                    # 1-indexed

# Compute CMIP6 matching year indices for each sample
cmip6_year_indices = [
    find_cmip6_year(month_indices_1[i], ΔTs[year_indices[i], month_indices_0[i]])
    for i in range(N_SAMPLES)
]


# =============================================================================
# PLOTTING
# =============================================================================

def collect_maps(var, year_indices, month_indices_1, cmip6_year_indices):
    """Collect 16 maps from each source for a variable."""
    maps_consistency = []
    maps_diffusion = []
    maps_cmip6 = []
    for i in range(N_SAMPLES):
        yr = int(year_indices[i])
        mo = int(month_indices_1[i])
        maps_consistency.append(
            piControl_consistency[var].isel(year=yr).sel(month=mo).values
        )
        maps_diffusion.append(
            piControl_diffusion[var].isel(year=yr).sel(month=mo).values
        )
        maps_cmip6.append(
            piControl_cmip6_anom[var].sel(month=mo).isel(year=cmip6_year_indices[i]).values
        )
    return (
        np.stack(maps_cmip6),          # (16, lat, lon)
        np.stack(maps_diffusion),      # (16, lat, lon)
        np.stack(maps_consistency),    # (16, lat, lon)
    )


def create_sample_plot(var):
    var_info = VARIABLES[var]
    cmap = var_info['cmap']
    unit = var_info['unit']
    var_name = var_info['name']

    maps_cmip6, maps_diffusion, maps_consistency = collect_maps(
        var, year_indices, month_indices_1, cmip6_year_indices
    )

    all_vals = np.concatenate([maps_cmip6, maps_diffusion, maps_consistency])
    vmax = np.quantile(np.abs(all_vals), 0.995)

    n_cols_per_source = 4
    n_rows = 4
    group_gap = 0.15   # fraction of a map width
    cbar_width = 0.06

    # width_ratios: [4 maps | gap | 4 maps | gap | 4 maps | cbar]
    width_ratios = (
        [1] * n_cols_per_source + [group_gap]
        + [1] * n_cols_per_source + [group_gap]
        + [1] * n_cols_per_source + [cbar_width]
    )
    height_ratios = [0.08] + [1] * n_rows

    nrows = len(height_ratios)
    ncols = len(width_ratios)
    fig = plt.figure(figsize=(2.2 * sum(width_ratios), 1.8 * sum(height_ratios)))
    gs = GridSpec(nrows, ncols, figure=fig,
                  width_ratios=width_ratios, height_ratios=height_ratios,
                  hspace=0.02, wspace=0.02)

    # Group title labels
    group_defs = [
        (0, f"{config.data.model_name}"),
        (5, "Diffusion"),
        (10, "Consistency"),
    ]
    for col_start, title in group_defs:
        ax_title = fig.add_subplot(gs[0, col_start:col_start + n_cols_per_source])
        ax_title.axis("off")
        ax_title.text(0.5, 0.2, title, ha='center', va='bottom',
                      fontsize=13, weight='bold', transform=ax_title.transAxes)

    source_list = [maps_cmip6, maps_diffusion, maps_consistency]
    col_offsets = [0, 5, 10]

    for src_idx, (maps, col_offset) in enumerate(zip(source_list, col_offsets)):
        for i in range(N_SAMPLES):
            row = (i // n_cols_per_source) + 1
            col = (i % n_cols_per_source) + col_offset
            ax = fig.add_subplot(gs[row, col], projection=ccrs.Robinson())
            ax.pcolormesh(
                piControl_cmip6_anom.lon.values,
                piControl_cmip6_anom.lat.values,
                maps[i],
                cmap=cmap,
                vmin=-vmax, vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading='auto',
                rasterized=True,
            )
            ax.coastlines(linewidth=0.3, alpha=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.2)

    # Shared colorbar on the right
    cax = fig.add_subplot(gs[1:, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical', extend='both')
    cbar.set_label(f"{var_name} anomaly [{unit}]", fontsize=10)

    fig.suptitle(f"{var_name} anomaly samples — piControl", fontsize=14, y=1.01)
    return fig


def main():
    for var in VARIABLES:
        print(f"Plotting samples for {var}...")
        fig = create_sample_plot(var)
        save_plot(fig, OUTPUT_DIR, f'samples_{var}.jpg', dpi=DPI)
        print(f"Saved samples_{var}.jpg")


if __name__ == "__main__":
    main()
