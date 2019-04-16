#!/bin/sh
#SBATCH --job-name=Infersent

#SBATCH --output=job_output_infersent-%j.txt

#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu

#SBATCH --time=0-12:00:00

#SBATCH --gres=gpu:p100:1


module load python/3.7.0

nvidia-smi

python3 infersent.py > infersent.out
