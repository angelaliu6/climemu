import os
import sys
import numpy as np
import xarray as xr
import jax.numpy as jnp
import pyshtools as pysh
import cartopy.crs as ccrs
from tqdm import tqdm

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
WSPACE = 0.01
HSPACE = 0.05
RANDOM_SEED = 5


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def compute_Cl(da):
    da = da - da.mean()
    grid = pysh.SHGrid.from_xarray(da, grid='GLQ')
    clm = grid.expand()
    Cl = clm.spectrum()
    return Cl

def get_plot_data(pred_da, cmip6_da):
    emulator_Cl = []
    cmip6_Cl = []
    n_year = len(cmip6_da.year)
    nt = 12 * n_year
    with tqdm(total=nt) as pbar:
        for yr in range(n_year):
            for m in range(12):
                Cl = compute_Cl(pred_da.isel(year=yr, month=m, drop=True))
                emulator_Cl.append(Cl)
                Cl = compute_Cl(cmip6_da.isel(year=yr, month=m, drop=True))
                cmip6_Cl.append(Cl)
                _ = pbar.update(1)
    emulator_Cl = jnp.stack(emulator_Cl)
    cmip6_Cl = jnp.stack(cmip6_Cl)

    n_boot = 10000
    rng = np.random.default_rng(None)
    boot_emulator = []
    boot_cmip6 = []
    for _ in range(n_boot):
        idx = rng.integers(0, nt, size=nt)
        boot_emulator.append(emulator_Cl[idx].mean(axis=0))
        boot_cmip6.append(cmip6_Cl[idx].mean(axis=0))
    boot_emulator = jnp.array(boot_emulator)
    boot_cmip6 = jnp.array(boot_cmip6)

    emulator = {'mean': jnp.mean(emulator_Cl, axis=0),
                'mean_ub': jnp.quantile(boot_emulator, 0.0275, axis=0),
                'mean_lb': jnp.quantile(boot_emulator, 0.975, axis=0),
                'lb': jnp.quantile(emulator_Cl, 0.0275, axis=0),
                'ub': jnp.quantile(emulator_Cl, 0.975, axis=0)}
    cmip6 = {'mean': jnp.mean(cmip6_Cl, axis=0),
             'mean_ub': jnp.quantile(boot_cmip6, 0.0275, axis=0),
             'mean_lb': jnp.quantile(boot_cmip6, 0.975, axis=0),
             'lb': jnp.quantile(cmip6_Cl, 0.0275, axis=0),
             'ub': jnp.quantile(cmip6_Cl, 0.975, axis=0)}
    return emulator, cmip6


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
climatology, piControl_consistency, piControl_cmip6 = load_data(config, in_memory=True)
piControl_cmip6 = piControl_cmip6 - climatology

piControl_diffusion = xr.open_dataset(os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_piControl.nc"))

R = 6371  # km
ell = np.arange(piControl_consistency.sizes['lat'])
k = np.where(ell > 0, ell, np.nan) / (2 * np.pi * R)

np.random.seed(RANDOM_SEED)
sample_year_idx = np.random.randint(len(piControl_consistency.year))
sample_month_idx = np.random.randint(12)
print(f"Sample for month {sample_month_idx + 1}")

consistency = dict()
diffusion = dict()
cmip6 = dict()

for i, var in enumerate(VARIABLES):
    print(f"Computing PSD for consistency model, {var}...")
    consistency_psd, cmip6_psd = get_plot_data(piControl_consistency[var], piControl_cmip6[var])
    consistency_psd['sample'] = piControl_consistency[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    cmip6_psd['sample'] = piControl_cmip6[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    consistency[var] = consistency_psd
    cmip6[var] = cmip6_psd

    print(f"Computing PSD for diffusion model, {var}...")
    diffusion_psd, _ = get_plot_data(piControl_diffusion[var], piControl_cmip6[var])
    diffusion_psd['sample'] = piControl_diffusion[var].isel(year=sample_year_idx, month=sample_month_idx, drop=True)
    diffusion[var] = diffusion_psd


# =============================================================================
# PLOTTING
# =============================================================================

def create_power_spectra_plot():
    width_ratios = [0.1, 1, 0.1, 0.25, 1, 0.1, 0.25, 1, 0.1, 0.25, 1, 0.1, 0.25]
    height_ratios = [0.8, 0.8, 1.5]

    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0.25, 0.5, f"{config.data.model_name} \n realization", va="center", ha="center", rotation="vertical", fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0.25, 0.5, "Consistency \n sample", va="center", ha="center", rotation="vertical", fontsize=12)

    for i, var in enumerate(VARIABLES):
        var_info = VARIABLES[var]
        unit = var_info['unit']
        cmap = var_info['cmap']
        col = 3 * i + 1
        flatvalues = []

        # CMIP6 sample
        ax = fig.add_subplot(gs[0, col], projection=ccrs.Robinson())
        mesh1 = cmip6[var]['sample'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        flatvalues.append(cmip6[var]['sample'].values.ravel())
        ax.coastlines(lw=0.5, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)
        ax.set_title(var_info['name'], fontsize=16, weight='bold')

        # Consistency sample
        ax = fig.add_subplot(gs[1, col], projection=ccrs.Robinson())
        mesh2 = consistency[var]['sample'].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
        flatvalues.append(consistency[var]['sample'].values.ravel())
        ax.coastlines(lw=0.5, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)

        vmax = np.quantile(np.concatenate(flatvalues), 0.999)
        vmin = np.quantile(np.concatenate(flatvalues), 0.001)
        vmax = max(np.abs(vmax), np.abs(vmin))
        mesh1.set_clim(-vmax, vmax)
        mesh2.set_clim(-vmax, vmax)

        cax = fig.add_subplot(gs[:2, col + 1])
        cbar = fig.colorbar(mesh1, cax=cax, orientation='vertical', extend='both')
        cbar.set_label(f"[{unit}]", labelpad=4)
        pos = cax.get_position()
        new_width = pos.width * 0.3
        new_height = pos.height * 0.3
        new_x0 = pos.x0 + 0.00
        new_y0 = pos.y0 + (pos.height - new_height) / 2
        cax.set_position([new_x0, new_y0, new_width, new_height])

        # Power spectra — all three models
        ax = fig.add_subplot(gs[2, col:col+2])
        ax.fill_between(k, consistency[var]['lb'], consistency[var]['ub'], color='tomato', alpha=0.15)
        ax.fill_between(k, diffusion[var]['lb'], diffusion[var]['ub'], color='darkorange', alpha=0.15)
        ax.fill_between(k, cmip6[var]['lb'], cmip6[var]['ub'], color='cornflowerblue', alpha=0.15)
        ax.plot(k, cmip6[var]['mean'], label=config.data.model_name, color='cornflowerblue')
        ax.plot(k, diffusion[var]['mean'], label='Diffusion', color='darkorange')
        ax.plot(k, consistency[var]['mean'], label='Consistency', color='tomato')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.margins(x=0, y=0)
        ax.set_xlabel('Inverse wavelength [km⁻¹]', fontsize=14)
        ax.set_ylabel(f'[({unit})²⋅km]', fontsize=14)
        if col == 1:
            ax.legend(fontsize=16, frameon=False)

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    fig = create_power_spectra_plot()
    save_plot(fig, OUTPUT_DIR, 'psd.jpg', dpi=DPI)

if __name__ == "__main__":
    main()
