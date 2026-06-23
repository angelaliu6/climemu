import copy
import os
from collections import deque
from functools import partial
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from src.utils.collate import numpy_collate
from src.diffusion import ct_make_step, ct_batch_loss, compute_bins, compute_ema_decay
from src.diffusion.losses.consistency_training import timesteps_to_times
from src.datasets import PatternToCMIP6Dataset
from paper.mpi.config import Config
from . import utils


@dataclass
class TrainingState:
    model: eqx.Module
    ema_model: eqx.Module
    opt_state: optax.OptState
    step: int = 0
    epoch: int = 0


def _ckpt_paths(base: str) -> dict:
    stem = base[:-4] if base.endswith('.eqx') else base
    return {
        'model': stem + '_model.eqx',
        'ema':   stem + '_ema.eqx',
        'opt':   stem + '_opt.eqx',
        'meta':  stem + '_meta.npz',
    }


def save_training_state(state: 'TrainingState', config: 'Config') -> None:
    paths = _ckpt_paths(config.training.checkpoint_filename)
    eqx.tree_serialise_leaves(paths['model'], state.model)
    eqx.tree_serialise_leaves(paths['ema'],   state.ema_model)
    eqx.tree_serialise_leaves(paths['opt'],   state.opt_state)
    np.savez(paths['meta'], step=np.array(state.step), epoch=np.array(state.epoch))
    print(f"Checkpoint saved at step {state.step}, epoch {state.epoch}")


def load_training_state(
    model: eqx.Module,
    ema_model: eqx.Module,
    opt_state: optax.OptState,
    config: 'Config',
) -> 'TrainingState | None':
    paths = _ckpt_paths(config.training.checkpoint_filename)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    model     = eqx.tree_deserialise_leaves(paths['model'], model)
    ema_model = eqx.tree_deserialise_leaves(paths['ema'],   ema_model)
    opt_state = eqx.tree_deserialise_leaves(paths['opt'],   opt_state)
    meta      = np.load(paths['meta'])
    step, epoch = int(meta['step']), int(meta['epoch'])
    print(f"Resuming from checkpoint: step={step}, epoch={epoch}")
    return TrainingState(model, ema_model, opt_state, step, epoch)


def log_training_metrics(state, loss, grad, mse, bins, ema_decay):
    wandb.log({
        "Train Loss": loss,
        "Gradient norm": grad,
        "Unweighted MSE": mse,
        "Bins": int(bins),
        "EMA decay": float(ema_decay),
    }, step=state.step)


def log_validation_metrics(config, state, val_loader, μ, σ, total_steps, χval):
    val_loss = 0
    n_val_steps = len(val_loader)
    with tqdm(total=n_val_steps, desc="Evaluation") as pbar:
        for batch_idx, batch in enumerate(val_loader):
            x = utils.process_batch(batch, μ, σ)
            _, χval = jr.split(χval)
            val_value, _ = ct_batch_loss(
                state.ema_model, state.ema_model,
                config.model.context_channels, x,
                jnp.array(state.step), jnp.array(total_steps),
                config.schedule.time_min, config.schedule.time_max,
                config.training.bins_min, config.training.bins_max,
                config.training.bins_rho, χval,
            )
            val_loss += val_value.item()
            pbar.set_description(f"Epoch {state.epoch + 1} | Val {round(val_loss / (batch_idx + 1), 2)}")
            pbar.update(1)
    wandb.log({"Validation Loss": val_loss / n_val_steps}, step=state.step)


def train_epoch(
    state: TrainingState,
    train_loader: DataLoader,
    val_loader: DataLoader,
    μ: jnp.ndarray,
    σ: jnp.ndarray,
    log_sampler: callable,
    log_target_data: jnp.ndarray,
    config: Config,
    optimizer: optax.GradientTransformation,
    total_steps: int,
) -> TrainingState:
    loss_queue = deque(maxlen=config.training.queue_length)
    grad_queue = deque(maxlen=config.training.queue_length)

    χtrain, χval = jr.split(jr.PRNGKey(state.epoch + 1))
    n_train_steps = len(train_loader)

    with tqdm(total=n_train_steps) as pbar:
        for batch in train_loader:
            x = utils.process_batch(batch, μ, σ)
            _, χtrain = jr.split(χtrain)

            step_arr = jnp.array(state.step)
            total_arr = jnp.array(total_steps)

            value, mse, model, χtrain, opt_state, grad_norm = ct_make_step(
                state.model, state.ema_model,
                config.model.context_channels, x,
                step_arr, total_arr,
                config.schedule.time_min, config.schedule.time_max,
                config.training.bins_min, config.training.bins_max,
                config.training.bins_rho,
                χtrain, state.opt_state, optimizer.update,
            )

            # Dynamic EMA decay (Philip's formula)
            ema_decay = compute_ema_decay(
                step_arr, total_arr,
                config.training.bins_min,
                config.training.initial_ema_decay,
            )
            ema_model = utils.update_ema(state.ema_model, model, ema_decay)
            state = TrainingState(model, ema_model, opt_state, state.step + 1, state.epoch)

            loss_queue.append(value.item())
            grad_queue.append(grad_norm.item())
            running_loss = sum(loss_queue) / len(loss_queue)
            running_grad = sum(grad_queue) / len(grad_queue)

            pbar.set_description(f"Epoch {state.epoch + 1} | Loss {round(running_loss, 2)}")
            pbar.update(1)

            if (state.step + 1) % config.training.log_interval == 0:
                bins = compute_bins(step_arr, total_arr, config.training.bins_min, config.training.bins_max)
                log_training_metrics(state, running_loss, running_grad, float(mse), bins, ema_decay)

            if (state.step + 1) % config.training.sample_interval == 0:
                log_validation_metrics(config, state, val_loader, μ, σ, total_steps, χval)
                _, χval = jr.split(χval)
                pred_samples = log_sampler(denoiser=ema_model, key=χtrain)
                utils.log_samples(pred_samples, log_target_data, config.data.variables, state.step)

    state = TrainingState(state.model, state.ema_model, state.opt_state, state.step, state.epoch + 1)
    if state.epoch % config.training.checkpoint_interval == 0:
        save_training_state(state, config)
    return state


def train(
    model: eqx.Module,
    train_dataset: PatternToCMIP6Dataset,
    val_dataset: PatternToCMIP6Dataset,
    schedule: object,
    μ: jnp.ndarray,
    σ: jnp.ndarray,
    config: Config,
) -> eqx.Module:
    optimizer = optax.chain(
        optax.clip_by_global_norm(50.0),
        optax.adam(learning_rate=config.training.learning_rate)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ema_model = copy.deepcopy(model)
    state = load_training_state(model, ema_model, opt_state, config) \
        or TrainingState(model, ema_model, opt_state)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        collate_fn=numpy_collate,
    )

    total_steps = config.training.epochs * len(train_loader)

    log_pattern, log_target_data = utils.get_sample_batch(
        dataset=train_dataset,
        batch_size=16,
        key=jr.PRNGKey(config.training.random_seed),
    )
    log_sampler = partial(
        utils.draw_samples_batch_consistency,
        schedule=schedule,
        pattern_batch=log_pattern,
        n_samples=config.training.sample_count,
        n_steps=config.training.sample_steps_consistency,
        μ=μ,
        σ=σ,
        output_size=(config.model.out_channels, config.model.input_size[1], config.model.input_size[2]),
        time_min=config.schedule.time_min,
        time_max=config.schedule.time_max,
        bins_rho=config.training.bins_rho,
    )

    wandb.init(project=config.training.wandb_project, config=config)
    utils.log_initial_context(log_pattern)

    for _ in range(state.epoch, config.training.epochs):
        state = train_epoch(
            state, train_loader, val_loader,
            μ, σ, log_sampler, log_target_data,
            config, optimizer, total_steps,
        )

    wandb.finish()
    return state.ema_model
