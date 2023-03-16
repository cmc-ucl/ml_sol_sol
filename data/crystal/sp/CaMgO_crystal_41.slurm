#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-gc-smw    
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4000
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4000


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4001
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4001


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4002
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4002


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4003
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4003


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4004
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4004


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4005
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4005


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4006
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4006


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4007
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4007


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4008
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4008


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4009
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4009


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4010
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4010


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4011
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4011


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4012
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4012


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4013
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4013


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4014
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4014


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4015
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4015


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4016
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4016


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4017
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4017


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4018
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4018


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4019
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4019


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4020
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4020


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4021
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4021


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4022
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4022


