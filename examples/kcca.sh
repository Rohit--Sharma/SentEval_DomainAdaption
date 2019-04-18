#!/bin/sh
#SBATCH --job-name=KCCA

#SBATCH --output=job_output_kcca-%j.txt

#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu

#SBATCH --time=0-12:00:00

#SBATCH --gres=gpu:titanrtx:1


module load python/3.7.0

nvidia-smi

python3 kcca.py > kcca.out
