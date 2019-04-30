#!/bin/sh
#SBATCH --job-name=DomainSpecificSentenceEmbeddings

#SBATCH --output=slurm_out/job_output_all-%j.txt

#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu

#SBATCH --time=0-12:00:00

#SBATCH --gres=gpu:1


module load python/3.7.0

nvidia-smi

python3 bert.py Amazon Yelp IMDB > results/bert.out
python3 cca.py Amazon Yelp IMDB > results/cca.out
python3 cnn.py Amazon Yelp IMDB > results/cnn.out
python3 concat.py Amazon Yelp IMDB > results/concat.out
python3 kcca.py Amazon Yelp IMDB > results/kcca.out
