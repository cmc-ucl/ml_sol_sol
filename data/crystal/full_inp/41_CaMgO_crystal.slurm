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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_637 CaMgO_full_637 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_636 CaMgO_full_636 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_635 CaMgO_full_635 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_634 CaMgO_full_634 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_633 CaMgO_full_633 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_632 CaMgO_full_632 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_631 CaMgO_full_631 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_630 CaMgO_full_630 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_629 CaMgO_full_629 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_628 CaMgO_full_628 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_627 CaMgO_full_627 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_625 CaMgO_full_625 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_613 CaMgO_full_613 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_624 CaMgO_full_624 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_623 CaMgO_full_623 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_622 CaMgO_full_622 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_621 CaMgO_full_621 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_620 CaMgO_full_620 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_619 CaMgO_full_619 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_618 CaMgO_full_618 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_617 CaMgO_full_617 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_616 CaMgO_full_616 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_615 CaMgO_full_615 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_614 CaMgO_full_614 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1847 CaMgO_full_1847 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3701 CaMgO_full_3701 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1849 CaMgO_full_1849 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3094 CaMgO_full_3094 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3092 CaMgO_full_3092 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3091 CaMgO_full_3091 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3090 CaMgO_full_3090 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3089 CaMgO_full_3089 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3088 CaMgO_full_3088 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3087 CaMgO_full_3087 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3086 CaMgO_full_3086 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3085 CaMgO_full_3085 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3084 CaMgO_full_3084 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3083 CaMgO_full_3083 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3082 CaMgO_full_3082 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3081 CaMgO_full_3081 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3080 CaMgO_full_3080 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3079 CaMgO_full_3079 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3078 CaMgO_full_3078 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3077 CaMgO_full_3077 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3076 CaMgO_full_3076 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3075 CaMgO_full_3075 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3074 CaMgO_full_3074 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3073 CaMgO_full_3073 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3072 CaMgO_full_3072 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3071 CaMgO_full_3071 &

wait
