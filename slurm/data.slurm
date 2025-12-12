#!/bin/bash -eux

#SBATCH --job-name=data
#SBATCH --mail-type=ALL
#SBATCH --account=sci-lippert
#SBATCH --mail-user=Masoumeh.Javanbakhat@hpi.de
#SBATCH --partition=gpu-batch # -p
#SBATCH --cpus-per-task=12 # -c
#SBATCH --gpus=v100:4
#SBATCH --mem=60gb
#SBATCH --export=ALL # exports all environment variables from the current shell environment to the job’s environment
#SBATCH --time=10:00:00
#SBATCH --error=/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub/adni_results/errors/torchLIMIX_resc_%j.err
#SBATCH --output=/sc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/AdniGithub/adni_results/logs/data5.log # %j is job id

source /sc/home/masoumeh.javanbakhat/miniconda3/etc/profile.d/conda.sh
conda activate myenv 


srun python ../src/data.py