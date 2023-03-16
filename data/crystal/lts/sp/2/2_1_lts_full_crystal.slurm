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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1875 lts_optgeom_full_1875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1881 lts_optgeom_full_1881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2010 lts_optgeom_full_2010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1883 lts_optgeom_full_1883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1988 lts_optgeom_full_1988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1986 lts_optgeom_full_1986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1976 lts_optgeom_full_1976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1969 lts_optgeom_full_1969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1962 lts_optgeom_full_1962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2301 lts_optgeom_full_2301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1958 lts_optgeom_full_1958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1956 lts_optgeom_full_1956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1952 lts_optgeom_full_1952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1947 lts_optgeom_full_1947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1945 lts_optgeom_full_1945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1926 lts_optgeom_full_1926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1925 lts_optgeom_full_1925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1886 lts_optgeom_full_1886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1885 lts_optgeom_full_1885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2289 lts_optgeom_full_2289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2343 lts_optgeom_full_2343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2302 lts_optgeom_full_2302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3129 lts_optgeom_full_3129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3063 lts_optgeom_full_3063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3033 lts_optgeom_full_3033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3029 lts_optgeom_full_3029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3026 lts_optgeom_full_3026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3025 lts_optgeom_full_3025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3015 lts_optgeom_full_3015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3013 lts_optgeom_full_3013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3010 lts_optgeom_full_3010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3009 lts_optgeom_full_3009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3008 lts_optgeom_full_3008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3006 lts_optgeom_full_3006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3005 lts_optgeom_full_3005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3003 lts_optgeom_full_3003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3001 lts_optgeom_full_3001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3000 lts_optgeom_full_3000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3115 lts_optgeom_full_3115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3185 lts_optgeom_full_3185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2995 lts_optgeom_full_2995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3186 lts_optgeom_full_3186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3605 lts_optgeom_full_3605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3603 lts_optgeom_full_3603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3591 lts_optgeom_full_3591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3565 lts_optgeom_full_3565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3561 lts_optgeom_full_3561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3537 lts_optgeom_full_3537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3521 lts_optgeom_full_3521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3287 lts_optgeom_full_3287 &

wait
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1875 lts_optgeom_full_1875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1881 lts_optgeom_full_1881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2010 lts_optgeom_full_2010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1883 lts_optgeom_full_1883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1988 lts_optgeom_full_1988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1986 lts_optgeom_full_1986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1976 lts_optgeom_full_1976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1969 lts_optgeom_full_1969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1962 lts_optgeom_full_1962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2301 lts_optgeom_full_2301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1958 lts_optgeom_full_1958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1956 lts_optgeom_full_1956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1952 lts_optgeom_full_1952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1947 lts_optgeom_full_1947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1945 lts_optgeom_full_1945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1926 lts_optgeom_full_1926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1925 lts_optgeom_full_1925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1886 lts_optgeom_full_1886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1885 lts_optgeom_full_1885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2289 lts_optgeom_full_2289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2343 lts_optgeom_full_2343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2302 lts_optgeom_full_2302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3129 lts_optgeom_full_3129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3063 lts_optgeom_full_3063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3033 lts_optgeom_full_3033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3029 lts_optgeom_full_3029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3026 lts_optgeom_full_3026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3025 lts_optgeom_full_3025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3015 lts_optgeom_full_3015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3013 lts_optgeom_full_3013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3010 lts_optgeom_full_3010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3009 lts_optgeom_full_3009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3008 lts_optgeom_full_3008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3006 lts_optgeom_full_3006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3005 lts_optgeom_full_3005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3003 lts_optgeom_full_3003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3001 lts_optgeom_full_3001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3000 lts_optgeom_full_3000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3115 lts_optgeom_full_3115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3185 lts_optgeom_full_3185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2995 lts_optgeom_full_2995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3186 lts_optgeom_full_3186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3605 lts_optgeom_full_3605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3603 lts_optgeom_full_3603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3591 lts_optgeom_full_3591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3565 lts_optgeom_full_3565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3561 lts_optgeom_full_3561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3537 lts_optgeom_full_3537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3521 lts_optgeom_full_3521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3287 lts_optgeom_full_3287 &

wait
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1875 lts_optgeom_full_1875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1881 lts_optgeom_full_1881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2010 lts_optgeom_full_2010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1883 lts_optgeom_full_1883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1988 lts_optgeom_full_1988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1986 lts_optgeom_full_1986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1976 lts_optgeom_full_1976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1969 lts_optgeom_full_1969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1962 lts_optgeom_full_1962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2301 lts_optgeom_full_2301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1958 lts_optgeom_full_1958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1956 lts_optgeom_full_1956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1952 lts_optgeom_full_1952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1947 lts_optgeom_full_1947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1945 lts_optgeom_full_1945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1926 lts_optgeom_full_1926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1925 lts_optgeom_full_1925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1886 lts_optgeom_full_1886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1885 lts_optgeom_full_1885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2289 lts_optgeom_full_2289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2343 lts_optgeom_full_2343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2302 lts_optgeom_full_2302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3129 lts_optgeom_full_3129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3063 lts_optgeom_full_3063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3033 lts_optgeom_full_3033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3029 lts_optgeom_full_3029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3026 lts_optgeom_full_3026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3025 lts_optgeom_full_3025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3015 lts_optgeom_full_3015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3013 lts_optgeom_full_3013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3010 lts_optgeom_full_3010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3009 lts_optgeom_full_3009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3008 lts_optgeom_full_3008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3006 lts_optgeom_full_3006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3005 lts_optgeom_full_3005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3003 lts_optgeom_full_3003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3001 lts_optgeom_full_3001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3000 lts_optgeom_full_3000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3115 lts_optgeom_full_3115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3185 lts_optgeom_full_3185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2995 lts_optgeom_full_2995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3186 lts_optgeom_full_3186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3605 lts_optgeom_full_3605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3603 lts_optgeom_full_3603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3591 lts_optgeom_full_3591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3565 lts_optgeom_full_3565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3561 lts_optgeom_full_3561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3537 lts_optgeom_full_3537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3521 lts_optgeom_full_3521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3287 lts_optgeom_full_3287 &

wait
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1875 lts_optgeom_full_1875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1881 lts_optgeom_full_1881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2010 lts_optgeom_full_2010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1883 lts_optgeom_full_1883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1988 lts_optgeom_full_1988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1986 lts_optgeom_full_1986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1976 lts_optgeom_full_1976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1969 lts_optgeom_full_1969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1962 lts_optgeom_full_1962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2301 lts_optgeom_full_2301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1958 lts_optgeom_full_1958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1956 lts_optgeom_full_1956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1952 lts_optgeom_full_1952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1947 lts_optgeom_full_1947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1945 lts_optgeom_full_1945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1926 lts_optgeom_full_1926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1925 lts_optgeom_full_1925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1886 lts_optgeom_full_1886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1885 lts_optgeom_full_1885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2289 lts_optgeom_full_2289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2343 lts_optgeom_full_2343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2302 lts_optgeom_full_2302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3129 lts_optgeom_full_3129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3063 lts_optgeom_full_3063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3033 lts_optgeom_full_3033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3029 lts_optgeom_full_3029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3026 lts_optgeom_full_3026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3025 lts_optgeom_full_3025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3015 lts_optgeom_full_3015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3013 lts_optgeom_full_3013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3010 lts_optgeom_full_3010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3009 lts_optgeom_full_3009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3008 lts_optgeom_full_3008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3006 lts_optgeom_full_3006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3005 lts_optgeom_full_3005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3003 lts_optgeom_full_3003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3001 lts_optgeom_full_3001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3000 lts_optgeom_full_3000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3115 lts_optgeom_full_3115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3185 lts_optgeom_full_3185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2995 lts_optgeom_full_2995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3186 lts_optgeom_full_3186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3605 lts_optgeom_full_3605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3603 lts_optgeom_full_3603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3591 lts_optgeom_full_3591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3565 lts_optgeom_full_3565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3561 lts_optgeom_full_3561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3537 lts_optgeom_full_3537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3521 lts_optgeom_full_3521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3287 lts_optgeom_full_3287 &

wait
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1875 lts_optgeom_full_1875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1881 lts_optgeom_full_1881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2010 lts_optgeom_full_2010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1883 lts_optgeom_full_1883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1988 lts_optgeom_full_1988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1986 lts_optgeom_full_1986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1976 lts_optgeom_full_1976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1969 lts_optgeom_full_1969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1962 lts_optgeom_full_1962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2301 lts_optgeom_full_2301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1958 lts_optgeom_full_1958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1956 lts_optgeom_full_1956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1952 lts_optgeom_full_1952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1947 lts_optgeom_full_1947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1945 lts_optgeom_full_1945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1926 lts_optgeom_full_1926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1925 lts_optgeom_full_1925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1886 lts_optgeom_full_1886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_1885 lts_optgeom_full_1885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2289 lts_optgeom_full_2289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2343 lts_optgeom_full_2343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2302 lts_optgeom_full_2302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3129 lts_optgeom_full_3129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3063 lts_optgeom_full_3063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3033 lts_optgeom_full_3033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3029 lts_optgeom_full_3029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3026 lts_optgeom_full_3026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3025 lts_optgeom_full_3025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3015 lts_optgeom_full_3015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3013 lts_optgeom_full_3013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3010 lts_optgeom_full_3010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3009 lts_optgeom_full_3009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3008 lts_optgeom_full_3008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3006 lts_optgeom_full_3006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3005 lts_optgeom_full_3005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3003 lts_optgeom_full_3003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3001 lts_optgeom_full_3001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3000 lts_optgeom_full_3000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3115 lts_optgeom_full_3115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3185 lts_optgeom_full_3185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_2995 lts_optgeom_full_2995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3186 lts_optgeom_full_3186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3605 lts_optgeom_full_3605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3603 lts_optgeom_full_3603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3591 lts_optgeom_full_3591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3565 lts_optgeom_full_3565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3561 lts_optgeom_full_3561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3537 lts_optgeom_full_3537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3521 lts_optgeom_full_3521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_optgeom_full_3287 lts_optgeom_full_3287 &

wait
