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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2633 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2604 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2634 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2659 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2658 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2657 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2656 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2655 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2654 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2653 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2652 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2651 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2650 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2649 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2648 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2647 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2646 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2645 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2644 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2643 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2642 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2641 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2640 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2639 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2638 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2637 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2636 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2635 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3759 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2574 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2572 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2571 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2570 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2569 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2568 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2567 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2566 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3757 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2563 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2562 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2560 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2559 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2558 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2557 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2556 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_2555 &

wait
