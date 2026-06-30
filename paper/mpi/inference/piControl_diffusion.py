import os
from typing import Iterator, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import xarray as xr
from functools import partial
from tqdm import tqdm

from src.diffusion import HealPIXUNet, ContinuousVESchedule

from ..config import Config
from ..data import load_dataset
from .. import utils


DIFFUSION_OUTPUT_DIR = "/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_diffusion"


def make_dataloader(n_year: int, β: jnp.ndarray, σpiControl: float, key: jr.PRNGKey, nlat: int, nlon: int) -> Iterator[jnp.ndarray]:
    for _ in range(n_year):
        key, _ = jr.split(key)
        ΔT = σpiControl * jr.normal(key, 12)
        context = β[..., 0] + β[..., 1] * ΔT.reshape(-1, 1)
        context = context.reshape(-1, nlat, nlon)
        yield context


def load_model_and_data(config: Config) -> Tuple:
    β = jnp.load(config.data.pattern_scaling_path)
    stats = jnp.load(config.data.norm_stats_path)
    μ_train, σ_train = jnp.array(stats['μ']), jnp.array(stats['σ'])

    test_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=["ssp245"],
        variables=["tas"],
        in_memory=False,
        external_β=β
    )
    lat = test_dataset.cmip6data.lat
    lon = test_dataset.cmip6data.lon
    σmax = float(np.load(config.data.sigma_max_path))

    edges_to_healpix, edges_to_latlon = utils.load_or_compute_edges(
        nside=config.model.nside,
        edges_path=config.model.edges_path
    )

    model = HealPIXUNet(
        input_size=config.model.input_size,
        nside=config.model.nside,
        enc_filters=list(config.model.enc_filters),
        dec_filters=list(config.model.dec_filters),
        out_channels=config.model.out_channels,
        temb_dim=config.model.temb_dim,
        healpix_emb_dim=config.model.healpix_emb_dim,
        edges_to_healpix=edges_to_healpix,
        edges_to_latlon=edges_to_latlon
    )
    model = eqx.tree_deserialise_leaves(config.training.model_filename, model)
    return model, β, lat, lon, μ_train, σ_train, σmax


def save_predictions(pred_samples: List[np.ndarray], lat, lon, output_dir: str, variables: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    numpy_array = np.concatenate(pred_samples, axis=1)
    n_year = numpy_array.shape[1]
    da = xr.DataArray(
        numpy_array,
        dims=["month", "year", "variable", "lat", "lon"],
        coords={
            "month": np.arange(1, 13),
            "year": np.arange(n_year),
            "variable": list(variables),
            "lat": lat,
            "lon": lon,
        }
    )
    ds = da.to_dataset(dim="variable")
    output_path = os.path.join(output_dir, "emulated_piControl.nc")
    ds.to_netcdf(output_path, mode='w')
    print(f"Predictions saved to {output_path}")


def main():
    config = Config()
    model, β, lat, lon, μ_train, σ_train, σmax = load_model_and_data(config)

    piControl = load_dataset(root=config.data.root_dir,
                             model=config.data.model_name,
                             experiments=["piControl"],
                             variables=["tas"],
                             in_memory=False,
                             external_β=β)
    σpiControl = piControl.gmst['piControl']['tas'].std().item()
    print("σpiControl = ", σpiControl)

    n_year = 1000
    key = jr.PRNGKey(0)
    nlat, nlon = config.model.input_size[1], config.model.input_size[2]
    test_loader = make_dataloader(n_year, β, σpiControl, key, nlat, nlon)

    schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
    χtest = jr.PRNGKey(config.sampling.random_seed)

    output_size = (config.model.out_channels, nlat, nlon)
    generate_samples = partial(utils.draw_samples_batch,
                               model=model,
                               schedule=schedule,
                               n_samples=1,
                               n_steps=config.sampling.n_steps,
                               μ=μ_train, σ=σ_train,
                               output_size=output_size)

    pred_samples = []
    with tqdm(total=n_year) as pbar:
        for batch_idx, pattern_batch in enumerate(test_loader):
            batch_pred_samples = generate_samples(pattern_batch=pattern_batch, key=χtest)
            pred_samples.append(jax.device_get(batch_pred_samples))
            χtest, _ = jr.split(χtest)

            if (batch_idx + 1) % 100 == 0:
                save_predictions(pred_samples, lat, lon, DIFFUSION_OUTPUT_DIR, config.data.variables)
            _ = pbar.update(1)

    save_predictions(pred_samples, lat, lon, DIFFUSION_OUTPUT_DIR, config.data.variables)


if __name__ == "__main__":
    main()
