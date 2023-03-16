#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-power-smw    
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4001 CaMgO_coord_4001
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4001


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4002 CaMgO_coord_4002
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4002


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4003 CaMgO_coord_4003
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4003


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4004 CaMgO_coord_4004
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4004


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4005 CaMgO_coord_4005
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4005


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4006 CaMgO_coord_4006
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4006


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4007 CaMgO_coord_4007
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4007


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4008 CaMgO_coord_4008
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4008


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4009 CaMgO_coord_4009
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4009


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4010 CaMgO_coord_4010
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4010


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4011 CaMgO_coord_4011
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4011


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4012 CaMgO_coord_4012
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4012


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4013 CaMgO_coord_4013
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4013


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4014 CaMgO_coord_4014
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4014


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4015 CaMgO_coord_4015
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4015


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4016 CaMgO_coord_4016
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4016


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4017 CaMgO_coord_4017
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4017


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4018 CaMgO_coord_4018
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4018


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4019 CaMgO_coord_4019
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4019


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4020 CaMgO_coord_4020
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4020


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4021 CaMgO_coord_4021
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4021


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4022 CaMgO_coord_4022
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4022


