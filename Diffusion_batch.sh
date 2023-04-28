#!/bin/bash
#SBATCH -J DDPM
#SBATCH -o DDPM.out
#SBATCH -G 1
#SBATCH --get-user-env
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 8


srun --mpi=pmix \
        --container-image=/users/rosenbaum/aheidar/projects/Diffusion/Diffusion.sqsh \
        --container-mounts=/users/rosenbaum/aheidar/projects/Diffusion:/mnt/Diffusion \
        /bin/bash -c "cd /mnt/Diffusion/
        python train.py --images_dir /mnt/Diffusion/data --par_refresh 1000 --num_iter 70000 --checkpoints_dir /mnt/Diffusion/checkpoints"
