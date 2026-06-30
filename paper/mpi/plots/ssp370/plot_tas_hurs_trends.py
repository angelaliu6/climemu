import os
import sys
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs

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
WIDTH_MULTIPLIER = 3.0
HEIGHT_MULTIPLIER = 2.0
WSPACE = 0.01
HSPACE = 0.01


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def load_land_fraction():
    land_fraction_filepath = "/orcd/data/raffaele/001/shahineb/products/cmip6/raw/MPI-ESM1-2-LR/piControl/r1i1p1f1/sftlf/sftlf_fx_MPI-ESM1-2-LR_piControl_r1i1p1f1_gn.nc"
    land_fraction_ds = xr.open_dataset(land_fraction_filepath)['sftlf']
    return land_fraction_ds


def create_regional_masks(land_fraction_ds, da_cmip6):
    land_fraction_ds = land_fraction_ds.interp(lat=da_cmip6.lat, lon=da_cmip6.lon, method='linear')
    land_mask = land_fraction_ds > 25
    land_mask = land_mask.where(land_mask.lat >= -60, 0)
    ocean_mask = land_fraction_ds < 25
    so_mask = ocean_mask.where(ocean_mask.lat < -55, 0)
    to_mask = ocean_mask.where((ocean_mask.lat < 10) & (ocean_mask.lat > -10), 0)
    arctic_mask = (land_mask.lat >= 80).astype(int).broadcast_like(land_mask)

    return {
        'land': land_mask,
        'southern_ocean': so_mask,
        'tropical_ocean': to_mask,
        'arctic': arctic_mask
    }


def process_region_data(da_cmip6, da_emulator, mask, wlat, region_name):
    if region_name == 'arctic':
        cmip6_data = da_cmip6.sel(lat=slice(80, 90)).mean(['lat', 'lon'], skipna=True).compute()
        emulator_data = da_emulator.sel(lat=slice(80, 90)).mean(['lat', 'lon'], skipna=True).compute()
    else:
        cmip6_data = da_cmip6.where(mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()
        emulator_data = da_emulator.where(mask == 1).weighted(wlat).mean(['lat', 'lon'], skipna=True).compute()

    ensemble_cmip6 = cmip6_data.groupby('time.year').mean()
    μ_cmip6 = cmip6_data.mean('member').groupby('time.year').mean()
    σ_cmip6 = cmip6_data.std('member').groupby('time.year').mean()
    ub_cmip6 = μ_cmip6 + 2 * σ_cmip6
    lb_cmip6 = μ_cmip6 - 2 * σ_cmip6

    μ_emulator = emulator_data.mean('member').groupby('time.year').mean()
    σ_emulator = emulator_data.std('member').groupby('time.year').mean()
    ub_emulator = μ_emulator + 2 * σ_emulator
    lb_emulator = μ_emulator - 2 * σ_emulator

    return {
        'ensemble_cmip6': ensemble_cmip6,
        'μ_cmip6': μ_cmip6,
        'ub_cmip6': ub_cmip6,
        'lb_cmip6': lb_cmip6,
        'μ_emulator': μ_emulator,
        'ub_emulator': ub_emulator,
        'lb_emulator': lb_emulator,
    }


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

da_cmip6 = xr.concat([target_data_hist[['tas', 'hurs']], target_data[['tas', 'hurs']]], dim='time')
da_consistency = xr.concat([pred_consistency_hist[['tas', 'hurs']], pred_consistency[['tas', 'hurs']]], dim='time')
da_diffusion = xr.concat([pred_diffusion_hist[['tas', 'hurs']], pred_diffusion[['tas', 'hurs']]], dim='time')

common_years = np.intersect1d(
    np.unique(da_cmip6.time.dt.year.values),
    np.unique(da_diffusion.time.dt.year.values)
)
da_cmip6 = da_cmip6.sel(time=da_cmip6.time.dt.year.isin(common_years))
da_consistency = da_consistency.sel(time=da_consistency.time.dt.year.isin(common_years))
da_diffusion = da_diffusion.sel(time=da_diffusion.time.dt.year.isin(common_years))

land_fraction_ds = load_land_fraction()
masks = create_regional_masks(land_fraction_ds, da_cmip6)
wlat = np.cos(np.deg2rad(da_cmip6.lat))

regions = ['land', 'southern_ocean', 'tropical_ocean', 'arctic']
region_data_consistency = {}
region_data_diffusion = {}
for region in regions:
    region_data_consistency[region] = process_region_data(da_cmip6, da_consistency, masks[region], wlat, region)
    region_data_diffusion[region] = process_region_data(da_cmip6, da_diffusion, masks[region], wlat, region)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_model_rows(fig, gs, region_data, row_offset, vars_list, time, colors, region_order, region_titles, vmin, vmax):
    for i, var in enumerate(vars_list):
        for j, (region, title) in enumerate(zip(region_order, region_titles)):
            ax = fig.add_subplot(gs[row_offset + i, j])
            data = region_data[region]
            color = colors[region]

            for ω in range(50):
                ax.plot(time, data['ensemble_cmip6'][var].isel(member=ω).values,
                       color="gray", lw=0.2, ls='--', alpha=0.2)

            ax.fill_between(time, data['lb_emulator'][var], data['ub_emulator'][var],
                           color=color, alpha=0.2)
            ax.plot(time, data['μ_emulator'][var], color=color, lw=1, alpha=1)

            if i == 0:
                if j == 0:
                    ax.set_ylabel("Temperature [°C]")
                    ax.set_yticks([0, 5, 10])
                else:
                    ax.yaxis.set_visible(False)
                if row_offset == 0:
                    ax.set_title(title, weight="bold")
            else:
                if j == 0:
                    ax.set_ylabel("Relative humidity [%]")
                    ax.set_yticks([-4, 0])
                else:
                    ax.yaxis.set_visible(False)

            ax.set_ylim(vmin[var], vmax[var])
            ax.set_xticks([1900, 2000])
            ax.margins(0)


def create_tas_hurs_trends_plot():
    all_values = []
    for region in regions:
        for data in [region_data_consistency[region], region_data_diffusion[region]]:
            all_values.extend([data['ub_cmip6'], data['lb_cmip6'], data['ub_emulator'], data['lb_emulator']])

    flat_values = xr.concat(all_values, dim='new')
    vmax = flat_values.quantile(q=0.99, dim=["new", "year"])
    vmin = flat_values.quantile(q=0.01, dim=["new", "year"])

    # 4 rows: consistency tas, consistency hurs, diffusion tas, diffusion hurs
    width_ratios = [1, 1, 1, 1, 0.05, 0.66]
    height_ratios = [1, 1, 1, 1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    time = common_years
    colors = {
        'land': "forestgreen",
        'arctic': "cornflowerblue",
        'southern_ocean': "b",
        'tropical_ocean': "palevioletred"
    }
    region_order = ['land', 'tropical_ocean', 'southern_ocean', 'arctic']
    region_titles = ['Land', 'Tropical Ocean', 'Southern Ocean', 'Arctic']

    # Consistency model rows (0–1)
    plot_model_rows(fig, gs, region_data_consistency, 0, ['tas', 'hurs'],
                    time, colors, region_order, region_titles, vmin, vmax)

    # Diffusion model rows (2–3)
    plot_model_rows(fig, gs, region_data_diffusion, 2, ['tas', 'hurs'],
                    time, colors, region_order, region_titles, vmin, vmax)

    # Row labels on the right
    for row_offset, label in [(0, "Consistency"), (2, "Diffusion")]:
        ax_label = fig.add_subplot(gs[row_offset:row_offset+2, 4])
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, label, va="center", ha="center",
                     rotation="vertical", fontsize=10, weight="bold")

    # Shared legend
    proxy_samples = Line2D([0], [0], color='0.3', ls='--', lw=0.5, alpha=0.5)
    proxy_mean = Line2D([0], [0], color='0.01', lw=1)
    proxy_fill = Patch(facecolor='0.3', alpha=0.4)

    fig.legend(
        handles=[proxy_samples, proxy_mean, proxy_fill],
        labels=[f"{config.data.model_name} members", "Emulated mean", "Emulator ±2σ"],
        loc='lower center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.45, -0.05)
    )

    # Regional map
    ax = fig.add_subplot(gs[:, -1], projection=ccrs.Robinson())
    ax.coastlines(linewidth=0.2, color="black")

    region_colors = {
        'land': "forestgreen",
        'arctic': "cornflowerblue",
        'southern_ocean': "b",
        'tropical_ocean': "palevioletred"
    }
    for region in region_order:
        color = region_colors[region]
        mask = masks[region]
        da1 = mask.where(mask == 1)
        cmap = ListedColormap(['none', color])
        ax.pcolormesh(mask.lon, mask.lat, da1,
                     transform=ccrs.PlateCarree(),
                     cmap=cmap, vmin=0, vmax=1, alpha=0.7, shading='auto')

    return fig


def main():
    fig = create_tas_hurs_trends_plot()
    save_plot(fig, OUTPUT_DIR, 'tas_hurs_trends.jpg', dpi=DPI)


if __name__ == "__main__":
    main()
