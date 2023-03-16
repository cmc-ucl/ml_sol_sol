#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --time=12:00:00
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=e05-power-smw
#SBATCH --partition=standard
#SBATCH --qos=standard   

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0
export OMP_NUM_THREADS=1

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3414 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3413 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3412 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3411 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3382 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3381 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3380 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3379 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3378 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3377 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3376 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3375 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3374 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3373 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3372 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3371 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3370 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3369 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3368 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3367 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3366 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3365 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3364 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3363 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3362 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3361 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3360 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3359 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3358 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3383 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3384 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3385 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3399 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3410 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3409 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3408 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3407 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3406 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3405 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3404 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3403 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3402 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3401 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3400 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3398 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3386 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3397 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3396 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3395 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3394 &

wait
