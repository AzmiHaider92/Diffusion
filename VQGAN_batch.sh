#!/bin/bash
#SBATCH -J AE_scheduler
#SBATCH -o AE_scheduler.out
#SBATCH -G 4
#SBATCH --get-user-env
#SBATCH --nodes 1
#SBATCH --time=6-24:00:00
#SBATCH --cpus-per-gpu 8


srun --mpi=pmix \
        --container-image=/users/rosenbaum/aheidar/projects/Diffusion/Diffusion.sqsh \
        --container-mounts=/users/rosenbaum/aheidar/projects/Diffusion:/mnt/Diffusion \
        /bin/bash -c "cd /mnt/Diffusion/
        source activate Diffusion
	python ldm/main.py --base "/mnt/Diffusion/ldm/configs/co3d_VQautoencoder.yaml""  
