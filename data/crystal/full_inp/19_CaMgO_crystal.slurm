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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1805 CaMgO_full_1805 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1804 CaMgO_full_1804 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1777 CaMgO_full_1777 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1775 CaMgO_full_1775 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1721 CaMgO_full_1721 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1774 CaMgO_full_1774 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1745 CaMgO_full_1745 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1744 CaMgO_full_1744 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1743 CaMgO_full_1743 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1742 CaMgO_full_1742 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1833 CaMgO_full_1833 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1741 CaMgO_full_1741 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1740 CaMgO_full_1740 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3853 CaMgO_full_3853 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3854 CaMgO_full_3854 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3855 CaMgO_full_3855 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1736 CaMgO_full_1736 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1735 CaMgO_full_1735 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1734 CaMgO_full_1734 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3856 CaMgO_full_3856 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1732 CaMgO_full_1732 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3857 CaMgO_full_3857 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1730 CaMgO_full_1730 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1729 CaMgO_full_1729 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1728 CaMgO_full_1728 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1727 CaMgO_full_1727 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3858 CaMgO_full_3858 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3859 CaMgO_full_3859 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1724 CaMgO_full_1724 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1723 CaMgO_full_1723 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1722 CaMgO_full_1722 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1746 CaMgO_full_1746 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1747 CaMgO_full_1747 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1748 CaMgO_full_1748 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1762 CaMgO_full_1762 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1773 CaMgO_full_1773 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1772 CaMgO_full_1772 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1771 CaMgO_full_1771 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_3852 CaMgO_full_3852 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1769 CaMgO_full_1769 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1768 CaMgO_full_1768 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1767 CaMgO_full_1767 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1766 CaMgO_full_1766 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1765 CaMgO_full_1765 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1764 CaMgO_full_1764 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1763 CaMgO_full_1763 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1761 CaMgO_full_1761 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1749 CaMgO_full_1749 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1760 CaMgO_full_1760 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_1759 CaMgO_full_1759 &

wait
