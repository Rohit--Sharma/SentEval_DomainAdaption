#!/bin/sh
#SBATCH --job-name=DomainSpecificSentenceEmbeddings
#SBATCH --output=slurm_out/job_output-%j.txt
#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1

module load python/3.7.0

nvidia-smi

N_EXPTS=10
CNN_EMBED="CNN_no_glove"
#CNN_EMBED="CNN_glove_non_trainable"
#CNN_EMBED="CNN_glove_trainable"
TASKS="Amazon Yelp IMDB"

#python3 bert.py $N_EXPTS $TASKS > results/bert.out
#python3 cnn.py $N_EXPTS $TASKS > results/cnn.out
python3 concat.py $N_EXPTS $CNN_EMBED $TASKS > results/concat_bert_$CNN_EMBED.out
python3 cca.py $N_EXPTS $CNN_EMBED $TASKS > results/cca_bert_$CNN_EMBED.out
python3 kcca.py $N_EXPTS $CNN_EMBED $TASKS > results/kcca_bert_$CNN_EMBED.out
