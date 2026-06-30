import os
import sys
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.piControl.utils import load_data, VARIABLES, setup_figure, save_plot

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = 'paper/mpi/plots/piControl/files'
DIFFUSION_OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_diffusion"
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.05
HSPACE = 0.05

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def compute_emd(foo, bar):
    emd = xr.apply_ufunc(
        wasserstein_distance,
        foo,
        bar,
        input_core_dims=[['flat'], ['flat']],
        exclude_dims={'flat'},
        vectorize=True,
        output_dtypes=[float],
        dask="parallelized")
    return emd

def add_seasonal_coords(data):
    month_to_season = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3:  "MAM", 4:  "MAM", 5:  "MAM",
        6:  "JJA", 7:  "JJA", 8:  "JJA",
        9:  "SON", 10: "SON", 11: "SON"
    }
    seasons = np.array([month_to_season[m] for m in data['month'].values])
    return data.assign_coords(season=("month", seasons))

def get_emd_data(emulator_data, cmip6_flat):
    emd = dict()
    emulator_flat = emulator_data.stack(flat=('year', 'month'))
    for var in VARIABLES.keys():
        emd_var = dict()
        emulator_flat_var = emulator_flat[var]
        cmip6_flat_var = cmip6_flat[var]
        for season in ["DJF", "MAM", "JJA", "SON"]:
            print(f"Computing EMD for {var} in {season}")
            emulator_season = emulator_flat_var.where(emulator_flat_var.season == season, drop=True)
            esm_data = cmip6_flat_var.where(cmip6_flat_var.season == season, drop=True)
            σesm = esm_data.std('flat')
            σesm = σesm.where(σesm > 0.1, 0.1)
            emd_var[season] = compute_emd(emulator_season, esm_data) / σesm
        emd[var] = emd_var
    return emd

# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
climatology, piControl_consistency, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology

diffusion_path = os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_piControl.nc")
piControl_diffusion = xr.open_dataset(diffusion_path)

# Add seasonal coordinates
piControl_consistency = add_seasonal_coords(piControl_consistency)
piControl_diffusion = add_seasonal_coords(piControl_diffusion)
piControl_cmip6 = add_seasonal_coords(piControl_cmip6)

# Flatten for EMD computation
piControl_cmip6_flat = piControl_cmip6.stack(flat=('year', 'month'))

print("Computing EMD for consistency model...")
emd_consistency = get_emd_data(piControl_consistency, piControl_cmip6_flat)
print("Computing EMD for diffusion model...")
emd_diffusion = get_emd_data(piControl_diffusion, piControl_cmip6_flat)

# Δemd = emd_diffusion - emd_consistency (positive = consistency is better)
delta_emd = {
    var: {season: emd_diffusion[var][season] - emd_consistency[var][season]
          for season in ["DJF", "MAM", "JJA", "SON"]}
    for var in VARIABLES.keys()
}

# =============================================================================
# PLOTTING
# =============================================================================

def plot_variable(fig, gs, var, i, vmax):
    var_info = VARIABLES[var]
    var_name = var_info['name']

    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, var_name, va="center", ha="center",
            rotation="vertical", fontsize=16, weight="bold")

    meshes = []
    for j, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        ax = fig.add_subplot(gs[i, j + 1], projection=ccrs.Robinson())
        mesh = delta_emd[var][season].plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap='RdBu', add_colorbar=False,
            vmin=-vmax, vmax=vmax
        )
        ax.coastlines()
        meshes.append(mesh)
        if i == 0:
            ax.set_title(f"{season}", fontsize=16, weight="bold")

    if i == 0:
        cax = fig.add_subplot(gs[1:-1, 5])
        cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', extend='both')
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticks([-vmax, 0, vmax])
        cbar.set_label("ΔEMD-to-noise ratio\n(diffusion − consistency)", labelpad=4, fontsize=14, weight="bold")


def create_emd_plot():
    # Compute symmetric color limits across all variables and seasons
    all_vals = [delta_emd[var][season].values for var in VARIABLES for season in ["DJF", "MAM", "JJA", "SON"]]
    vmax = float(np.nanquantile(np.abs(np.concatenate([v.ravel() for v in all_vals])), 0.98))

    width_ratios = [0.05, 1, 1, 1, 1, 0.05]
    height_ratios = [1, 1, 1, 1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    plot_variable(fig, gs, 'tas', 0, vmax)
    plot_variable(fig, gs, 'pr', 1, vmax)
    plot_variable(fig, gs, 'hurs', 2, vmax)
    plot_variable(fig, gs, 'sfcWind', 3, vmax)

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    fig = create_emd_plot()
    save_plot(fig, OUTPUT_DIR, 'emd.jpg', dpi=DPI)

if __name__ == "__main__":
    main()
