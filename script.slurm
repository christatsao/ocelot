#! /bin/bash
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH -o ./logs/xcryonet_run_slurm-%j.out.%N 
#SBATCH -e ./logs/xcryonet_run_slurm-%j.err.%N 
#SBATCH --ntasks=4 
#SBATCH --account=soc-gpu-np 
#SBATCH --partition=soc-gpu-np
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=12GB

/uufs/chpc.utah.edu/common/home/u6052852/anaconda3/envs/myenv/bin/python train.py -resample 1 -lr 0.005 -wd 0.001





 
