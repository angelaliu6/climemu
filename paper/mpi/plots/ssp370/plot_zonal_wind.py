import os
import sys
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.ssp370.utils import load_data, setup_figure, save_plot
from paper.mpi.plots.historical.utils import load_data as load_historical_data


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/ssp370/files'
DIFFUSION_OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_diffusion"
DPI = 300
WIDTH_MULTIPLIER = 5.0
HEIGHT_MULTIPLIER = 2.5
WSPACE = 0.01
HSPACE = 0.15


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def process_zonal_wind_data(da_cmip6, da_emulator):
    σ_cmip6 = da_cmip6.mean('lon').std('member').groupby('time.year').mean().compute()

    da_cmip6_mean = da_cmip6.mean(['member', 'lon']).groupby('time.year').mean().compute()
    da_emulator_mean = da_emulator.mean(['member', 'lon']).groupby('time.year').mean().compute()

    common_years = np.intersect1d(da_cmip6_mean['year'].values, da_emulator_mean['year'].values)
    da_cmip6_mean = da_cmip6_mean.sel(year=common_years)
    da_emulator_mean = da_emulator_mean.sel(year=common_years)
    σ_cmip6 = σ_cmip6.sel(year=common_years)

    cmip6_vals = da_cmip6_mean.values.T
    emulator_vals = da_emulator_mean.values.T
    σ_cmip6_vals = σ_cmip6.where(σ_cmip6 > 0.1, 0.1).values.T
    bnr = np.abs(emulator_vals - cmip6_vals) / σ_cmip6_vals

    lats = da_cmip6_mean['lat'].values
    years = common_years
    Year, Lat = np.meshgrid(years, lats)

    return cmip6_vals, emulator_vals, bnr, Year, Lat, lats, years


def compute_contour_levels(cmip6_vals, emulator_vals, bnr):
    flat_values = np.concatenate([cmip6_vals, emulator_vals])
    vmax = np.quantile(flat_values, 0.99)
    vmin = np.quantile(flat_values, 0.01)
    vmax = max(np.abs(vmax), np.abs(vmin))
    locator = ticker.MaxNLocator(nbins=29, prune=None)
    levels = locator.tick_values(-vmax, vmax)

    bnrmax = max(1, np.quantile(bnr, 0.99))
    locator = ticker.MaxNLocator(nbins=14, prune=None)
    bnrlevels = locator.tick_values(0, bnrmax)

    return levels, bnrlevels


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_consistency, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp370'].ds

test_dataset_hist, pred_consistency_hist, _, __ = load_historical_data(config, in_memory=False)
target_data_hist = test_dataset_hist['historical'].ds

pred_diffusion = xr.open_dataset(os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_ssp370.nc"), chunks={})
pred_diffusion_hist = xr.open_dataset(os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_historical.nc"), chunks={})

da_cmip6 = xr.concat([target_data_hist['sfcWind'], target_data['sfcWind']], dim='time')
da_consistency = xr.concat([pred_consistency_hist['sfcWind'], pred_consistency['sfcWind']], dim='time')
da_diffusion = xr.concat([pred_diffusion_hist['sfcWind'], pred_diffusion['sfcWind']], dim='time')

common_years = np.intersect1d(
    np.unique(da_cmip6.time.dt.year.values),
    np.unique(da_diffusion.time.dt.year.values)
)
da_cmip6 = da_cmip6.sel(time=da_cmip6.time.dt.year.isin(common_years))
da_consistency = da_consistency.sel(time=da_consistency.time.dt.year.isin(common_years))
da_diffusion = da_diffusion.sel(time=da_diffusion.time.dt.year.isin(common_years))

cmip6_vals, consistency_vals, bnr_consistency, Year, Lat, lats, years = process_zonal_wind_data(da_cmip6, da_consistency)
_, diffusion_vals, bnr_diffusion, _, _, _, _ = process_zonal_wind_data(da_cmip6, da_diffusion)

all_emulator_vals = np.concatenate([consistency_vals, diffusion_vals])
all_bnr = np.concatenate([bnr_consistency, bnr_diffusion])
levels, bnrlevels = compute_contour_levels(cmip6_vals, all_emulator_vals, all_bnr)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_zonal_wind_row(fig, gs, row, cmip6_v, emulator_v, bnr_v, levels, bnrlevels, label, show_ylabel=True):
    yticks = [-60, -30, 0, 30, 60]
    ytick_labels = ["60S", "30S", "0", "30N", "60N"]

    ax1 = fig.add_subplot(gs[row, 0])
    c1 = ax1.contourf(Year, Lat, cmip6_v, levels=levels, extend='both', cmap='PRGn')
    ax1.set_title(config.data.model_name, weight="bold")
    if show_ylabel:
        ax1.set_ylabel("Latitude")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels)
    ax1.set_xticks([1900, 2000, 2100])
    ax1.set_xticklabels([1900, 2000, 2100])

    ax2 = fig.add_subplot(gs[row, 1])
    ax2.contourf(Year, Lat, emulator_v, levels=levels, extend='both', cmap='PRGn')
    ax2.set_title(f"Emulator ({label})", weight="bold")
    ax2.set_yticks([])
    ax2.set_xticks([1900, 2000, 2100])
    ax2.set_xticklabels([1900, 2000, 2100])

    cax = fig.add_subplot(gs[row, 3])
    cbar = fig.colorbar(c1, cax=cax, orientation="vertical", shrink=0.8, pad=0.05)
    cbar.ax.set_yticks([-0.5, 0, 0.5])
    if show_ylabel:
        cbar.set_label("Windspeed anomaly [m/s]")

    ax3 = fig.add_subplot(gs[row, 5])
    c3 = ax3.contourf(Year, Lat, bnr_v, levels=bnrlevels, extend='max', cmap='RdPu')
    ax3.set_title("Error-to-noise ratio", weight="bold")
    ax3.set_yticks([])
    ax3.set_xticks([1900, 2000, 2100])
    ax3.set_xticklabels([1900, 2000, 2100])

    cax2 = fig.add_subplot(gs[row, 7])
    cbar2 = fig.colorbar(c3, cax=cax2, orientation="vertical", shrink=0.8, pad=0.05)
    cbar2.ax.set_yticks([0, 1])
    if show_ylabel:
        cbar2.set_label("[1]", labelpad=-1)


def create_zonal_wind_plot():
    width_ratios = [1, 1, 0.02, 0.05, 0.2, 1, 0.02, 0.05]
    height_ratios = [1, 0.08, 1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    plot_zonal_wind_row(fig, gs, 0, cmip6_vals, consistency_vals, bnr_consistency,
                        levels, bnrlevels, "Consistency", show_ylabel=True)
    plot_zonal_wind_row(fig, gs, 2, cmip6_vals, diffusion_vals, bnr_diffusion,
                        levels, bnrlevels, "Diffusion", show_ylabel=True)

    return fig


def main():
    fig = create_zonal_wind_plot()
    save_plot(fig, OUTPUT_DIR, 'westerlies.jpg', dpi=DPI)


if __name__ == "__main__":
    main()
