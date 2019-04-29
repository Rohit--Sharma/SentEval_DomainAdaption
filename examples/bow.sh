#!/bin/sh
#SBATCH --job-name=BOW

#SBATCH --output=job_output_bow-%j.txt

#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu

#SBATCH --time=0-12:00:00

#SBATCH --gres=gpu:1


module load python/3.7.0

nvidia-smi

python3 bow.py > bow.out
