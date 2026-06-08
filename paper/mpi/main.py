import os
import jax.numpy as jnp
import equinox as eqx

from src.diffusion import HealPIXUNet, ContinuousVESchedule
from .config import Config
from .data import load_dataset, compute_normalization, estimate_sigma_max
from .trainer import train
from .utils import load_or_compute_edges, print_parameter_count



def main():
    """Main entry point for training the climate diffusion model.
    
    This function:
    1. Loads configuration
    2. Prepares training and validation datasets
    3. Computes normalization statistics + maximum noise level
    4. Sets up the diffusion process and model
    5. Trains the model
    6. Saves the trained model
    """
    # Load configuration
    config = Config()

    # Load training dataset with pattern scaling
    train_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=config.data.train_experiments,
        variables=config.data.variables,
        in_memory=config.data.in_memory,
        pattern_scaling_path=config.data.pattern_scaling_path
    )
    
    # Load validation dataset using pattern scaling from training
    # This ensures consistent pattern scaling between train and validation
    val_dataset = load_dataset(
        root=config.data.root_dir,
        model=config.data.model_name,
        experiments=config.data.val_experiments,
        variables=config.data.variables,
        subset={'time': slice(*config.data.val_time_slice)},
        in_memory=config.data.in_memory,
        external_β=train_dataset.β  # Use training pattern scaling coefficients
    )
    
    # Compute normalization statistics from a random subset of the training data
    μ_train, σ_train = compute_normalization(
        train_dataset,
        config.training.batch_size,
        max_samples=config.data.norm_max_samples,
        seed=config.training.random_seed,
        norm_stats_path=config.data.norm_stats_path
    )

    # Estimate sigma_max for the dataset (if not already cached)
    sigma_max = estimate_sigma_max(
        dataset=train_dataset,
        μ=μ_train,
        σ=σ_train,
        ctx_size=config.model.context_channels,
        search_interval=config.data.sigma_max_search_interval,
        seed=config.training.random_seed,
        sigma_max_path=config.data.sigma_max_path
    )

    # Setup noise schedule for the diffusion process
    if config.schedule.sigma_max:
        sigma_max = config.schedule.sigma_max
    schedule = ContinuousVESchedule(config.schedule.sigma_min, sigma_max)

    # Load or compute Latlon-HEALPix edges
    edges_to_healpix, edges_to_latlon = load_or_compute_edges(
        nside=config.model.nside,
        lat=train_dataset.cmip6data.lat,
        lon=train_dataset.cmip6data.lon,
        edges_path=config.model.edges_path
    )

    # Initialize the UNet model with pre-trained diffusion weights
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
    print_parameter_count(model)
    model = eqx.tree_deserialise_leaves(config.training.model_filename, model)

    # Initialize denoiser with preconditioning
    σ2 = σ_train[:-1]**2
    σdata2 = σ2.mean()
    σdata = jnp.sqrt(σdata2)  # 2.307688
    print("Using σdata = ", σdata)
    σmin = config.schedule.sigma_min

    @eqx.filter_jit
    def c_skip(σ):
        return σdata2 / ((σ - σmin)**2 + σdata2)

    @eqx.filter_jit
    def c_out(σ):
        return σdata * (σ - σmin) / jnp.sqrt(σdata2 + σ**2)

    class Denoiser(eqx.Module):
        unet: HealPIXUNet
        ctx_size: int = eqx.field(static=True)
        def __call__(self, x, σ):
            return c_skip(σ) * (1 + σ) * x[:-self.ctx_size] + c_out(σ) * self.unet(x, σ)

    denoiser = Denoiser(model, config.model.context_channels)
        
    # Train the model
    denoiser = train(denoiser, train_dataset, val_dataset, schedule, μ_train, σ_train, config)
    
    # Save the trained model
    EXPERIMENT_DIR = os.path.dirname(__file__)
    CACHE_DIR = os.path.join(EXPERIMENT_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(CACHE_DIR, "weights_consistency.eqx"), denoiser) ## for diffusion, config.training.model_filename
    print(f"Model saved to weights_consistency.eqx")    ## for diffusion, config.training.model_filename


if __name__ == "__main__":
    main()