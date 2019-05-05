#!/bin/sh
#SBATCH --job-name=DomainSpecificSentenceEmbeddings
#SBATCH --output=slurm_out/job_output_InferSent-%j.txt
#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1

module load python/3.7.0

nvidia-smi

N_EXPTS=10
CNN_EMBED="CNN_no_glove"
# CNN_EMBED="CNN_glove_non_trainable"
# CNN_EMBED="CNN_glove_trainable"
TASKS="Amazon Yelp IMDB"

python3 infersent_cnn.py $N_EXPTS $CNN_EMBED $TASKS > results/infersent_cca_$CNN_EMBED.out
