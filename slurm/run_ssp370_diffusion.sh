#!/bin/bash
#SBATCH --job-name=ssp370_diffusion
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal_gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/angela5/climemu/slurm/logs/ssp370_diffusion_%j.out
#SBATCH --error=/home/angela5/climemu/slurm/logs/ssp370_diffusion_%j.err

cd /home/angela5/climemu
/home/angela5/.conda/envs/climemu_env/bin/python -m paper.mpi.inference.ssp370_diffusion
