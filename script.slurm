#! /bin/bash
#SBATCH --time=36:00:00 
#SBATCH --nodes=1
#SBATCH -o xcryonet_run_slurm-%j.out.%N 
#SBATCH -e xcryonet_run_slurm-%j.err.%N 
#SBATCH --ntasks=8 
#SBATCH --account=soc-gpu-kp 
#SBATCH --partition=soc-gpu-kp 
#SBATCH --gres=gpu:p100:1 
#SBATCH --mem=50000

/uufs/chpc.utah.edu/common/home/u6052852/anaconda3/envs/myenv/bin/python train.py -lr 0.001
/uufs/chpc.utah.edu/common/home/u6052852/anaconda3/envs/myenv/bin/python train.py -lr 0.005
/uufs/chpc.utah.edu/common/home/u6052852/anaconda3/envs/myenv/bin/python train.py -lr 0.009
 
