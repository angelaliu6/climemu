import os
import sys
import numpy as np
import regionmask
from tqdm import tqdm
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import colors
from scipy.stats import wasserstein_distance
import matplotlib.cm as cm
import matplotlib.colors as mcolors

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from paper.mpi.config import Config
from paper.mpi.plots.ssp245.utils import VARIABLES, load_data, setup_figure, save_plot

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/ssp245/files'
DIFFUSION_OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_diffusion"
DPI = 300
WIDTH_MULTIPLIER = 0.25
HEIGHT_MULTIPLIER = 1.0
WSPACE = 0.2
HSPACE = 0.1

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def setup_regions():
    ar6 = regionmask.defined_regions.ar6.all
    north_america = ar6[[0, 1, 2, 3, 4, 5, 6, 8]]
    south_america = ar6[[7, 9, 10, 11, 12, 13, 14, 15]]
    europe = ar6[[16, 17, 18, 19]]
    africa = ar6[[20, 21, 22, 23, 24, 25, 26, 27]]
    asia = ar6[[28, 29, 30, 31, 32, 33, 34, 35, 36, 37]]
    se_asia_oceania = ar6[[38, 39, 40, 41, 42, 43]]
    poles = ar6[[44, 45, 46]]
    oceans = ar6[[47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]]

    domains = [north_america, south_america, europe, africa, asia, se_asia_oceania, poles, oceans]
    domain_names = ["North America", "South America", "Europe", "Africa", "Asia", "Australasia", "Polar", "Oceans"]

    abbrevs = []
    for d in domains:
        abbrevs += list(map(lambda x: x.abbrev, d.regions.values()))

    return ar6, domains, domain_names, abbrevs


def compute_portrait_data(target_data, pred_data, periods, seasons, ar6, domains, domain_names):
    n_iterations = len(periods) * len(VARIABLES) * len(ar6) * len(seasons)
    portrait_data = np.zeros((len(periods), len(VARIABLES), len(ar6), 4))

    with tqdm(total=n_iterations) as pbar:
        for i, period in enumerate(periods):
            target_data_i = target_data.sel(time=period).load()
            pred_data_i = pred_data.sel(time=period).load()
            for j, var in enumerate(VARIABLES.keys()):
                target_data_ij = target_data_i[var]
                pred_data_ij = pred_data_i[var]
                k = 0
                for domain_idx, (domain, domain_name) in enumerate(zip(domains, domain_names)):
                    domain_mask = domain.mask(target_data.lon, target_data.lat)
                    for region_idx, region in domain.regions.items():
                        region_mask = domain_mask == region_idx
                        target_data_ijk = target_data_ij.where(region_mask)
                        pred_data_ijk = pred_data_ij.where(region_mask)
                        for l, season in enumerate(seasons):
                            pbar.set_description(f"Processing {period.start}-{period.stop}, {var}, {domain_name}, {region.name}, {season}")
                            target_data_ijkl = target_data_ijk.sel(time=target_data_ijk['time.season'] == season).values.flatten()
                            pred_data_ijkl = pred_data_ijk.sel(time=pred_data_ijk['time.season'] == season).values.flatten()
                            target_data_ijkl = target_data_ijkl[~np.isnan(target_data_ijkl)]
                            pred_data_ijkl = pred_data_ijkl[~np.isnan(pred_data_ijkl)]
                            emd = wasserstein_distance(target_data_ijkl, pred_data_ijkl).item()
                            σtarget = target_data_ijkl.std()
                            σ = max(σtarget, 0.1)
                            portrait_data[i, j, k, l] = emd / σ
                            _ = pbar.update(1)
                        k += 1

    return portrait_data

# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_data_consistency, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds

import xarray as xr
pred_data_diffusion = xr.open_dataset(os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_ssp245.nc"), chunks={})

ar6, domains, domain_names, abbrevs = setup_regions()
periods = [slice("2040-01", "2060-12"), slice("2080-01", "2100-12")]
seasons = ["DJF", "MAM", "JJA", "SON"]

print("Computing portrait data for consistency model...")
portrait_consistency = compute_portrait_data(target_data, pred_data_consistency, periods, seasons, ar6, domains, domain_names)
print("Computing portrait data for diffusion model...")
portrait_diffusion = compute_portrait_data(target_data, pred_data_diffusion, periods, seasons, ar6, domains, domain_names)

# Δportrait = diffusion - consistency (positive = consistency is better)
delta_portrait = portrait_diffusion - portrait_consistency

# =============================================================================
# PLOTTING
# =============================================================================

def portrait_plot(fig, ax, seasonal_data, y_labels, vmax):
    n_rows, n_cols, n_seasons = seasonal_data.shape
    patches = []
    colors_flat = []
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            vals = seasonal_data[i, j]

            x0, x1 = j,   j+1
            y0, y1 = i,   i+1
            xm, ym = (x0+x1)/2, (y0+y1)/2

            corners = [
                [(x0,y0),(x1,y0),(xm,ym)],  # DJF
                [(x1,y0),(x1,y1),(xm,ym)],  # MAM
                [(x1,y1),(x0,y1),(xm,ym)],  # JJA
                [(x0,y1),(x0,y0),(xm,ym)],  # SON
            ]

            for k, tri in enumerate(corners):
                patches.append(Polygon(tri))
                colors_flat.append(vals[k])

    coll = PatchCollection(patches, array=np.array(colors_flat),
                           cmap='RdBu', norm=norm, edgecolor='grey', linewidth=0.2)
    ax.add_collection(coll)

    ax.set_yticks(np.arange(n_rows)+0.5)
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    return fig, ax


def create_season_legend(fig, cax):
    bb = cax.get_position()
    width = bb.width * 3
    height = 7 * width
    x0 = bb.x0  + (bb.width - width) / 3
    y0 = bb.y0 - height*1.2

    ax_legend = fig.add_axes([x0, y0, width, height])
    ax_legend.axis("off")

    sq = Rectangle((0, 0), 1, 1, fill=False, lw=1.0, transform=ax_legend.transAxes)
    ax_legend.add_patch(sq)

    ax_legend.plot([0,1], [0,1], transform=ax_legend.transAxes, color='k', lw=0.5)
    ax_legend.plot([0,1], [1,0], transform=ax_legend.transAxes, color='k', lw=0.5)
    ax_legend.text(0.5, 0.1, "DJF", ha="center", va="bottom",
                   transform=ax_legend.transAxes, fontsize=6)
    ax_legend.text(0.58, 0.5, "MAM", ha="left", va="center",
                   transform=ax_legend.transAxes, fontsize=6)
    ax_legend.text(0.5, 0.85, "JJA", ha="center", va="top",
                   transform=ax_legend.transAxes, fontsize=6)
    ax_legend.text(0.4, 0.5, "SON", ha="right", va="center",
                   transform=ax_legend.transAxes, fontsize=6)


def create_portrait_plot():
    vmax = float(np.quantile(np.abs(delta_portrait), 0.98))

    width_ratios = [1] + [1] * len(abbrevs) + [1]
    height_ratios = [0.05, 1, 0.05, 1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    x_labels = abbrevs
    y_labels = list(VARIABLES.keys())

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "2040-2060", va="center", ha="center",
            rotation="vertical", fontsize=8, weight="bold")

    ax = fig.add_subplot(gs[3, 0])
    ax.axis("off")
    ax.text(0.5, 0.5, "2080-2100", va="center", ha="center",
            rotation="vertical", fontsize=8, weight="bold")

    idx = 2 + np.cumsum(list(map(len, domains)))
    idx = [3] + idx.tolist()
    for i, dname in enumerate(domain_names):
        ax = fig.add_subplot(gs[0, idx[i]:idx[i + 1]])
        ax.axis("off")
        ax.text(0.5, 0.5, dname, ha="center", fontsize=8, weight="bold")

    ax = fig.add_subplot(gs[1, 1:])
    fig, ax = portrait_plot(fig, ax, delta_portrait[0], y_labels, vmax)
    ax.get_xaxis().set_visible(False)

    ax = fig.add_subplot(gs[3, 1:])
    fig, ax = portrait_plot(fig, ax, delta_portrait[1], y_labels, vmax)
    n_rows, n_cols, n_seasons = delta_portrait[1].shape
    ax.set_xticks(np.arange(n_cols)+0.5)
    ax.set_xticklabels(x_labels, rotation=90, ha='center')

    cax = fig.add_subplot(gs[1:, -2])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sm = cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, extend="both")
    cbar.ax.set_yticks([-vmax, 0, vmax])
    cbar.set_label('ΔEMD-to-noise ratio\n(diffusion − consistency)')

    create_season_legend(fig, cax)

    return fig


def main():
    fig = create_portrait_plot()
    save_plot(fig, OUTPUT_DIR, 'portrait.jpg', dpi=DPI)


if __name__ == "__main__":
    main()
