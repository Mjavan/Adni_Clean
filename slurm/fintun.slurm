#!/bin/bash -eux


#SBATCH --job-name=finetun_data
#SBATCH --mail-type=ALL
#SBATCH --account=sci-lippert
#SBATCH --mail-user=Masoumeh.Javanbakhat@hpi.de
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=1 # -c
#SBATCH --gpus=1
#SBATCH --mem=10g
#SBATCH --time=10:00:00
#SBATCH --output=/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub/adni_results/logs/fintun_uncr1.log # %j is job id


srun python ../src/data_fintune.py