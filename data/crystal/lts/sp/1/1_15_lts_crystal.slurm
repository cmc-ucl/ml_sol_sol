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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1404 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1403 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1402 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1401 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1400 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1399 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1398 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1397 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1396 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1395 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3879 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1393 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_931 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1391 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1390 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1389 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1388 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1387 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1386 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3880 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1410 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1411 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1412 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1426 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1437 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3872 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1435 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3873 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1433 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1432 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1431 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1430 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1429 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1428 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3874 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1425 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1413 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1423 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_3876 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1421 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1420 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1419 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1418 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1417 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1416 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1415 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1414 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1495 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi lts_sp_1497 &

wait
