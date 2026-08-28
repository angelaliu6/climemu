"""One-off diagnostic: does per-step time/memory scale linearly with n_steps?

Answers the question raised while debugging compute_benchmark.py's >60x CPU
speedup for consistency (1-step) vs. diffusion (60 NFE / 30 Heun steps): is
this a memory-capacity effect (diffusion holding a (state, drift) pair vs.
consistency's state-only) or a per-step latency effect? Reuses the already-
loaded models/schedule/pattern from compute_benchmark.py (importing it does
not run its __main__ benchmark loop, only the module-level model loading).

Not part of the paper pipeline -- scratch diagnostic, safe to delete after use.
"""
import os
import time
import resource
import numpy as np
import jax

from paper.mpi.plots.compute_benchmark import (
    diffusion_model, consistency_model, schedule, pattern_batch,
    μ_train, σ_train, output_size, χ, config, σmax, _pytree_to_device,
)
from paper.mpi import utils

N_RUNS = 8
DIFFUSION_STEPS = [2, 5, 10, 20, 30]  # Heun/StepTo needs >=2 timesteps
CONSISTENCY_STEPS = [1, 5, 10, 30, 60]


def _rss_mb():
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _time_calls(fn, n_runs=N_RUNS):
    out = fn()
    jax.block_until_ready(out)
    rss_before = _rss_mb()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    rss_after = _rss_mb()
    times = np.array(times)
    return times.mean(), times.std(), rss_after - rss_before, rss_after


def main():
    cpu_device = jax.devices('cpu')[0]

    # eqx.Module pytrees (model, schedule) carry non-array cached leaves (e.g. a
    # compiled function on HealPIXUNet) that plain jax.device_put chokes on --
    # reuse the same helper compute_benchmark.py already needed for this.
    diffusion_cpu = _pytree_to_device(diffusion_model, cpu_device)
    consistency_cpu = _pytree_to_device(consistency_model, cpu_device)
    schedule_cpu = _pytree_to_device(schedule, cpu_device)
    pattern_batch_cpu = jax.device_put(pattern_batch, cpu_device)
    μ_cpu = jax.device_put(μ_train, cpu_device)
    σ_cpu = jax.device_put(σ_train, cpu_device)
    χ_cpu = jax.device_put(χ, cpu_device)

    print("=== Diffusion (Heun sampler): time & RSS vs. n_steps ===")
    print(f"{'n_steps':>8} {'NFE':>5} {'mean_s':>10} {'std_s':>10} {'d_rss_MB':>10} {'peak_rss_MB':>12} {'s_per_NFE':>10}")
    for n_steps in DIFFUSION_STEPS:
        fn = lambda: utils.draw_samples_batch(
            model=diffusion_cpu, schedule=schedule_cpu, pattern_batch=pattern_batch_cpu,
            n_samples=1, n_steps=n_steps, μ=μ_cpu, σ=σ_cpu,
            output_size=output_size, key=χ_cpu,
        )
        mean, std, d_rss, peak_rss = _time_calls(fn)
        nfe = 2 * n_steps
        print(f"{n_steps:>8} {nfe:>5} {mean:>10.4f} {std:>10.4f} {d_rss:>10.2f} {peak_rss:>12.1f} {mean/nfe:>10.5f}", flush=True)

    print("\n=== Consistency sampler: time & RSS vs. n_steps ===")
    print(f"{'n_steps':>8} {'NFE':>5} {'mean_s':>10} {'std_s':>10} {'d_rss_MB':>10} {'peak_rss_MB':>12} {'s_per_NFE':>10}")
    for n_steps in CONSISTENCY_STEPS:
        fn = lambda: utils.draw_samples_batch_consistency(
            denoiser=consistency_cpu, schedule=schedule_cpu, pattern_batch=pattern_batch_cpu,
            n_samples=1, n_steps=n_steps, μ=μ_cpu, σ=σ_cpu,
            output_size=output_size,
            time_min=config.schedule.time_min, time_max=float(σmax),
            bins_rho=config.training.bins_rho, key=χ_cpu,
        )
        mean, std, d_rss, peak_rss = _time_calls(fn)
        nfe = n_steps
        print(f"{n_steps:>8} {nfe:>5} {mean:>10.4f} {std:>10.4f} {d_rss:>10.2f} {peak_rss:>12.1f} {mean/nfe:>10.5f}", flush=True)


if __name__ == "__main__":
    main()
