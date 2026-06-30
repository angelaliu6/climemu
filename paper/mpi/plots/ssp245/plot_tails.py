import os
import sys
import numpy as np
import xarray as xr
import seaborn as sns

base_dir = os.path.join(os.getcwd())
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.utils import arrays
from paper.mpi.config import Config
from paper.mpi.plots.ssp245.utils import load_data, VARIABLES, setup_figure, save_plot
from paper.mpi.plots.piControl.utils import load_data as load_piControl_data


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'paper/mpi/plots/ssp245/files'
DIFFUSION_OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_diffusion"
DPI = 300
WIDTH_MULTIPLIER = 4.0
HEIGHT_MULTIPLIER = 3.0
WSPACE = 0.1
HSPACE = 0.3


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def process_tail_data(data, var_name):
    data_values = data[var_name].values.ravel()
    if var_name == "tas":
        data_values = data_values - 273.15
    return data_values


def compute_tail_quantiles(data):
    return np.quantile(data, [0.99, 0.999, 0.9999, 0.99999])


def create_tail_histogram(ax, data, bins, color, alpha, label):
    sns.histplot(data, ax=ax, kde=False, element="step", stat="density",
                fill=False, bins=bins, color=color, alpha=alpha, label=label)


def add_quantile_axes(ax, quantiles, color, label_prefix):
    sec_axis = ax.secondary_xaxis('bottom')
    sec_axis.set_xticks(quantiles)
    sec_axis.set_xticklabels([f"{label_prefix}%", f"{label_prefix}.9%",
                             f"{label_prefix}.99%", f"{label_prefix}.999%"],
                            color=color, ha='left', fontsize=10, rotation=45)
    sec_axis.xaxis.set_ticks_position('top')
    sec_axis.spines['bottom'].set_color(color)
    sec_axis.tick_params(axis='x', colors=color, pad=0.1)


# =============================================================================
# DATA LOADING
# =============================================================================

config = Config()
test_dataset, pred_consistency, _, __ = load_data(config, in_memory=False)
target_data = test_dataset['ssp245'].ds.sel(time=slice('2080-01', '2100-12'))
pred_consistency = pred_consistency.sel(time=slice('2080-01', '2100-12'))

pred_diffusion = xr.open_dataset(os.path.join(DIFFUSION_OUTPUT_DIR, "emulated_ssp245.nc"), chunks={})
pred_diffusion = pred_diffusion.sel(time=slice('2080-01', '2100-12'))

climatology, _, piControl_cmip6 = load_piControl_data(config, in_memory=False)

target_data = arrays.groupby_month_and_year(target_data) + climatology
pred_consistency = arrays.groupby_month_and_year(pred_consistency) + climatology
pred_diffusion = arrays.groupby_month_and_year(pred_diffusion) + climatology

target_data = target_data.compute()
pred_consistency = pred_consistency.compute()
pred_diffusion = pred_diffusion.compute()


# =============================================================================
# PLOTTING
# =============================================================================

def create_tail_plot():
    width_ratios = [1, 1, 1, 1]
    height_ratios = [1]
    fig, gs = setup_figure(width_ratios, height_ratios, WIDTH_MULTIPLIER, HEIGHT_MULTIPLIER, WSPACE, HSPACE)

    for i, var in enumerate(VARIABLES.keys()):
        var_info = VARIABLES[var]
        var_name = var_info["name"]
        unit = var_info['unit']

        cmip6_data = process_tail_data(target_data, var)
        consistency_data = process_tail_data(pred_consistency, var)
        diffusion_data = process_tail_data(pred_diffusion, var)

        if var == "tas":
            unit = "°C"

        qcmip6 = compute_tail_quantiles(cmip6_data)
        qconsistency = compute_tail_quantiles(consistency_data)
        qdiffusion = compute_tail_quantiles(diffusion_data)
        γ = min(qcmip6[0], qconsistency[0], qdiffusion[0]) - 0.1
        cmip6_tail = cmip6_data[cmip6_data >= γ]
        consistency_tail = consistency_data[consistency_data >= γ]
        diffusion_tail = diffusion_data[diffusion_data >= γ]

        bins_cmip6 = np.histogram_bin_edges(cmip6_tail, bins="fd")
        bins_consistency = np.histogram_bin_edges(consistency_tail, bins="fd")
        bins_diffusion = np.histogram_bin_edges(diffusion_tail, bins="fd")

        ax = fig.add_subplot(gs[0, i])
        create_tail_histogram(ax, cmip6_tail, bins_cmip6, "dodgerblue", 0.8, f"{config.data.model_name}")
        create_tail_histogram(ax, diffusion_tail, bins_diffusion, "darkorange", 0.8, "Diffusion")
        create_tail_histogram(ax, consistency_tail, bins_consistency, "tomato", 0.8, "Consistency")

        ax.set_yscale('log')
        ax.set_xlabel(f"{var_name} [{unit}]")
        if i > 0:
            ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.margins(x=0, y=0)
        if i == 3:
            ax.legend(loc='upper right')

        add_quantile_axes(ax, qcmip6, "dodgerblue", "99")
        add_quantile_axes(ax, qdiffusion, "darkorange", "99")
        add_quantile_axes(ax, qconsistency, "tomato", "99")

    return fig


def main():
    fig = create_tail_plot()
    save_plot(fig, OUTPUT_DIR, 'tails.jpg', dpi=DPI)


if __name__ == "__main__":
    main()
