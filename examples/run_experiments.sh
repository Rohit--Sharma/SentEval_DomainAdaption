#!/bin/sh
#SBATCH --job-name=DomainSpecificSentenceEmbeddings

#SBATCH --output=slurm_out/job_output_all-%j.txt

#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu

#SBATCH --time=0-12:00:00

#SBATCH --gres=gpu:1


module load python/3.7.0

nvidia-smi

python3 bert.py > bert.out
python3 cca.py > cca.out
python3 cnn.py > becnnrt.out
python3 concat.py > concat.out
python3 kcca.py > kcca.out
