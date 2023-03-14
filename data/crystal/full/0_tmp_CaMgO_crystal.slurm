#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --time=12:00:00
#SBATCH --nodes=100
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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3987 CaMgO_full_tmp_3987 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3988 CaMgO_full_tmp_3988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3990 CaMgO_full_tmp_3990 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3991 CaMgO_full_tmp_3991 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3992 CaMgO_full_tmp_3992 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3993 CaMgO_full_tmp_3993 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3994 CaMgO_full_tmp_3994 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3995 CaMgO_full_tmp_3995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3996 CaMgO_full_tmp_3996 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3997 CaMgO_full_tmp_3997 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3998 CaMgO_full_tmp_3998 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3999 CaMgO_full_tmp_3999 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4000 CaMgO_full_tmp_4000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4001 CaMgO_full_tmp_4001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4002 CaMgO_full_tmp_4002 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4003 CaMgO_full_tmp_4003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4004 CaMgO_full_tmp_4004 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4005 CaMgO_full_tmp_4005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4006 CaMgO_full_tmp_4006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4007 CaMgO_full_tmp_4007 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4008 CaMgO_full_tmp_4008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4009 CaMgO_full_tmp_4009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4010 CaMgO_full_tmp_4010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4011 CaMgO_full_tmp_4011 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4012 CaMgO_full_tmp_4012 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4013 CaMgO_full_tmp_4013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4014 CaMgO_full_tmp_4014 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4015 CaMgO_full_tmp_4015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4016 CaMgO_full_tmp_4016 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4017 CaMgO_full_tmp_4017 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4018 CaMgO_full_tmp_4018 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4019 CaMgO_full_tmp_4019 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4020 CaMgO_full_tmp_4020 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4021 CaMgO_full_tmp_4021 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4022 CaMgO_full_tmp_4022 &

wait
cp CaMgO_full_tmp_3987.f9 CaMgO_full_3987.f9
cp CaMgO_full_tmp_3987.gui CaMgO_full_3987.gui
cp CaMgO_full_tmp_3988.f9 CaMgO_full_3988.f9
cp CaMgO_full_tmp_3988.gui CaMgO_full_3988.gui
cp CaMgO_full_tmp_3990.f9 CaMgO_full_3990.f9
cp CaMgO_full_tmp_3990.gui CaMgO_full_3990.gui
cp CaMgO_full_tmp_3991.f9 CaMgO_full_3991.f9
cp CaMgO_full_tmp_3991.gui CaMgO_full_3991.gui
cp CaMgO_full_tmp_3992.f9 CaMgO_full_3992.f9
cp CaMgO_full_tmp_3992.gui CaMgO_full_3992.gui
cp CaMgO_full_tmp_3993.f9 CaMgO_full_3993.f9
cp CaMgO_full_tmp_3993.gui CaMgO_full_3993.gui
cp CaMgO_full_tmp_3994.f9 CaMgO_full_3994.f9
cp CaMgO_full_tmp_3994.gui CaMgO_full_3994.gui
cp CaMgO_full_tmp_3995.f9 CaMgO_full_3995.f9
cp CaMgO_full_tmp_3995.gui CaMgO_full_3995.gui
cp CaMgO_full_tmp_3996.f9 CaMgO_full_3996.f9
cp CaMgO_full_tmp_3996.gui CaMgO_full_3996.gui
cp CaMgO_full_tmp_3997.f9 CaMgO_full_3997.f9
cp CaMgO_full_tmp_3997.gui CaMgO_full_3997.gui
cp CaMgO_full_tmp_3998.f9 CaMgO_full_3998.f9
cp CaMgO_full_tmp_3998.gui CaMgO_full_3998.gui
cp CaMgO_full_tmp_3999.f9 CaMgO_full_3999.f9
cp CaMgO_full_tmp_3999.gui CaMgO_full_3999.gui
cp CaMgO_full_tmp_4000.f9 CaMgO_full_4000.f9
cp CaMgO_full_tmp_4000.gui CaMgO_full_4000.gui
cp CaMgO_full_tmp_4001.f9 CaMgO_full_4001.f9
cp CaMgO_full_tmp_4001.gui CaMgO_full_4001.gui
cp CaMgO_full_tmp_4002.f9 CaMgO_full_4002.f9
cp CaMgO_full_tmp_4002.gui CaMgO_full_4002.gui
cp CaMgO_full_tmp_4003.f9 CaMgO_full_4003.f9
cp CaMgO_full_tmp_4003.gui CaMgO_full_4003.gui
cp CaMgO_full_tmp_4004.f9 CaMgO_full_4004.f9
cp CaMgO_full_tmp_4004.gui CaMgO_full_4004.gui
cp CaMgO_full_tmp_4005.f9 CaMgO_full_4005.f9
cp CaMgO_full_tmp_4005.gui CaMgO_full_4005.gui
cp CaMgO_full_tmp_4006.f9 CaMgO_full_4006.f9
cp CaMgO_full_tmp_4006.gui CaMgO_full_4006.gui
cp CaMgO_full_tmp_4007.f9 CaMgO_full_4007.f9
cp CaMgO_full_tmp_4007.gui CaMgO_full_4007.gui
cp CaMgO_full_tmp_4008.f9 CaMgO_full_4008.f9
cp CaMgO_full_tmp_4008.gui CaMgO_full_4008.gui
cp CaMgO_full_tmp_4009.f9 CaMgO_full_4009.f9
cp CaMgO_full_tmp_4009.gui CaMgO_full_4009.gui
cp CaMgO_full_tmp_4010.f9 CaMgO_full_4010.f9
cp CaMgO_full_tmp_4010.gui CaMgO_full_4010.gui
cp CaMgO_full_tmp_4011.f9 CaMgO_full_4011.f9
cp CaMgO_full_tmp_4011.gui CaMgO_full_4011.gui
cp CaMgO_full_tmp_4012.f9 CaMgO_full_4012.f9
cp CaMgO_full_tmp_4012.gui CaMgO_full_4012.gui
cp CaMgO_full_tmp_4013.f9 CaMgO_full_4013.f9
cp CaMgO_full_tmp_4013.gui CaMgO_full_4013.gui
cp CaMgO_full_tmp_4014.f9 CaMgO_full_4014.f9
cp CaMgO_full_tmp_4014.gui CaMgO_full_4014.gui
cp CaMgO_full_tmp_4015.f9 CaMgO_full_4015.f9
cp CaMgO_full_tmp_4015.gui CaMgO_full_4015.gui
cp CaMgO_full_tmp_4016.f9 CaMgO_full_4016.f9
cp CaMgO_full_tmp_4016.gui CaMgO_full_4016.gui
cp CaMgO_full_tmp_4017.f9 CaMgO_full_4017.f9
cp CaMgO_full_tmp_4017.gui CaMgO_full_4017.gui
cp CaMgO_full_tmp_4018.f9 CaMgO_full_4018.f9
cp CaMgO_full_tmp_4018.gui CaMgO_full_4018.gui
cp CaMgO_full_tmp_4019.f9 CaMgO_full_4019.f9
cp CaMgO_full_tmp_4019.gui CaMgO_full_4019.gui
cp CaMgO_full_tmp_4020.f9 CaMgO_full_4020.f9
cp CaMgO_full_tmp_4020.gui CaMgO_full_4020.gui
cp CaMgO_full_tmp_4021.f9 CaMgO_full_4021.f9
cp CaMgO_full_tmp_4021.gui CaMgO_full_4021.gui
cp CaMgO_full_tmp_4022.f9 CaMgO_full_4022.f9
cp CaMgO_full_tmp_4022.gui CaMgO_full_4022.gui
