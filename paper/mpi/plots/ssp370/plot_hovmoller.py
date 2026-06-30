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
WIDTH_MULTIPLIER = 4.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.1
HSPACE = 0.15


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def process_hovmoller_data(da_cmip6, da_emulator):
    σ_cmip6 = da_cmip6.sel(lat=slice(-30, 30)).mean('lat').std('member').groupby('time.year').mean().compute()

    da_cmip6_mean = da_cmip6.sel(lat=slice(-30, 30)).mean(['member', 'lat']).groupby('time.year').mean().compute()
    da_emulator_mean = da_emulator.sel(lat=slice(-30, 30)).mean(['member', 'lat']).groupby('time.year').mean().compute()

    common_years = np.intersect1d(da_cmip6_mean['year'].values, da_emulator_mean['year'].values)
    da_cmip6_mean = da_cmip6_mean.sel(year=common_years)
    da_emulator_mean = da_emulator_mean.sel(year=common_years)
    σ_cmip6 = σ_cmip6.sel(year=common_years)

    lons = da_cmip6_mean['lon'].values
    lons = ((lons + 180) % 360) - 180
    sort_idx = np.argsort(lons)
    lons = lons[sort_idx]
    years = common_years

    cmip6_vals = da_cmip6_mean.values[:, sort_idx]
    emulator_vals = da_emulator_mean.values[:, sort_idx]
    σ_cmip6_vals = σ_cmip6.values[:, sort_idx]
    bnr = np.abs(emulator_vals - cmip6_vals) / σ_cmip6_vals

    Lon, Year = np.meshgrid(lons, years)
    return cmip6_vals, emulator_vals, bnr, Lon, Year, lons, years


def compute_contour_levels(cmip6_vals, emulator_vals, bnr):
    flat_values = np.concatenate([cmip6_vals, emulator_vals])
    vmax = np.quantile(flat_values, 0.99)
    vmin = np.quantile(flat_values, 0.01)
    vmax = max(np.abs(vmax), np.abs(vmin))
    locator = ticker.MaxNLocator(nbins=14, prune=None)
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

da_cmip6 = xr.concat([target_data_hist['pr'], target_data['pr']], dim='time')
da_consistency = xr.concat([pred_consistency_hist['pr'], pred_consistency['pr']], dim='time')
da_diffusion = xr.concat([pred_diffusion_hist['pr'], pred_diffusion['pr']], dim='time')

common_years = np.intersect1d(
    np.unique(da_cmip6.time.dt.year.values),
    np.unique(da_diffusion.time.dt.year.values)
)
da_cmip6 = da_cmip6.sel(time=da_cmip6.time.dt.year.isin(common_years))
da_consistency = da_consistency.sel(time=da_consistency.time.dt.year.isin(common_years))
da_diffusion = da_diffusion.sel(time=da_diffusion.time.dt.year.isin(common_years))

cmip6_vals, consistency_vals, bnr_consistency, Lon, Year, lons, years = process_hovmoller_data(da_cmip6, da_consistency)
_, diffusion_vals, bnr_diffusion, _, _, _, _ = process_hovmoller_data(da_cmip6, da_diffusion)

# Use shared contour levels across both models
all_emulator_vals = np.concatenate([consistency_vals, diffusion_vals])
all_bnr = np.concatenate([bnr_consistency, bnr_diffusion])
levels, bnrlevels = compute_contour_levels(cmip6_vals, all_emulator_vals, all_bnr)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_hovmoller_row(fig, gs, row, cmip6_v, emulator_v, bnr_v, levels, bnrlevels, label, show_yticks=True):
    xticks = [-120, -60, 0, 60, 120]
    xtick_labels = ["120W", "60W", "0", "60E", "120E"]

    ax1 = fig.add_subplot(gs[row, 0])
    c1 = ax1.contourf(Lon, Year, cmip6_v, levels=levels, extend='both', cmap='BrBG')
    ax1.set_title(config.data.model_name, weight="bold")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    if show_yticks:
        ax1.set_yticks([1900, 2000, 2100])
    else:
        ax1.set_yticks([])
    ax1.set_ylabel(label, fontsize=10, weight="bold")

    ax2 = fig.add_subplot(gs[row, 1])
    ax2.contourf(Lon, Year, emulator_v, levels=levels, extend='both', cmap='BrBG')
    ax2.set_title(f"Emulator ({label})", weight="bold")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels)
    ax2.yaxis.set_visible(False)

    cax = fig.add_subplot(gs[row, 2])
    cb = fig.colorbar(c1, cax=cax)
    cb.set_ticks([-0.25, 0, 0.25])
    if show_yticks:
        cb.set_label("Precipitation anomaly [mm/day]")

    ax3 = fig.add_subplot(gs[row, 4])
    c3 = ax3.contourf(Lon, Year, bnr_v, levels=bnrlevels, extend='max', cmap='RdPu')
    ax3.set_title("Error-to-noise ratio", weight="bold")
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xtick_labels)
    ax3.yaxis.set_visible(False)

    cax2 = fig.add_subplot(gs[row, 5])
    cb2 = fig.colorbar(c3, cax=cax2)
    cb2.ax.set_yticks([0, 1])
    if show_yticks:
        cb2.set_label("[1]")


def create_hovmoller_plot():
    width_ratios = [1, 1, 0.05, 0.3, 1, 0.05]
    height_ratios = [1, 0.08, 1]  # row 1 = consistency, thin spacer, row 2 = diffusion
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    plot_hovmoller_row(fig, gs, 0, cmip6_vals, consistency_vals, bnr_consistency,
                       levels, bnrlevels, "Consistency", show_yticks=True)
    plot_hovmoller_row(fig, gs, 2, cmip6_vals, diffusion_vals, bnr_diffusion,
                       levels, bnrlevels, "Diffusion", show_yticks=True)

    return fig


def main():
    fig = create_hovmoller_plot()
    save_plot(fig, OUTPUT_DIR, 'hovmoller.jpg', dpi=DPI)


if __name__ == "__main__":
    main()
