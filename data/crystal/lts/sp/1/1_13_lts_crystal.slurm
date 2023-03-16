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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1727 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3858 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3859 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1724 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1723 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1722 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1746 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1747 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1748 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1762 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1773 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1772 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1771 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3852 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1769 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1768 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1767 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1766 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1765 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1764 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1763 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1761 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1749 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1760 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1759 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1758 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1757 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1756 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1755 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1754 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1753 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1752 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1751 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1750 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1612 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1611 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1610 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1440 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1466 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3866 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1464 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1463 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1462 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1461 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1460 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3867 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1458 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3868 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1456 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1455 &

wait
