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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2400 CaMgO_full_tmp_2400 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2401 CaMgO_full_tmp_2401 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2402 CaMgO_full_tmp_2402 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2403 CaMgO_full_tmp_2403 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2404 CaMgO_full_tmp_2404 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2405 CaMgO_full_tmp_2405 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2406 CaMgO_full_tmp_2406 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2407 CaMgO_full_tmp_2407 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2408 CaMgO_full_tmp_2408 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2409 CaMgO_full_tmp_2409 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2410 CaMgO_full_tmp_2410 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2411 CaMgO_full_tmp_2411 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2412 CaMgO_full_tmp_2412 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2413 CaMgO_full_tmp_2413 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2414 CaMgO_full_tmp_2414 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2415 CaMgO_full_tmp_2415 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2416 CaMgO_full_tmp_2416 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2417 CaMgO_full_tmp_2417 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2418 CaMgO_full_tmp_2418 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2419 CaMgO_full_tmp_2419 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2420 CaMgO_full_tmp_2420 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2421 CaMgO_full_tmp_2421 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2422 CaMgO_full_tmp_2422 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2423 CaMgO_full_tmp_2423 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2424 CaMgO_full_tmp_2424 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2425 CaMgO_full_tmp_2425 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2426 CaMgO_full_tmp_2426 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2427 CaMgO_full_tmp_2427 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2428 CaMgO_full_tmp_2428 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2429 CaMgO_full_tmp_2429 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2430 CaMgO_full_tmp_2430 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2431 CaMgO_full_tmp_2431 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2432 CaMgO_full_tmp_2432 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2433 CaMgO_full_tmp_2433 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2434 CaMgO_full_tmp_2434 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2435 CaMgO_full_tmp_2435 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2436 CaMgO_full_tmp_2436 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2437 CaMgO_full_tmp_2437 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2438 CaMgO_full_tmp_2438 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2439 CaMgO_full_tmp_2439 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2440 CaMgO_full_tmp_2440 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2441 CaMgO_full_tmp_2441 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2442 CaMgO_full_tmp_2442 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2443 CaMgO_full_tmp_2443 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2444 CaMgO_full_tmp_2444 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2445 CaMgO_full_tmp_2445 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2446 CaMgO_full_tmp_2446 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2447 CaMgO_full_tmp_2447 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2448 CaMgO_full_tmp_2448 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2449 CaMgO_full_tmp_2449 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2450 CaMgO_full_tmp_2450 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2451 CaMgO_full_tmp_2451 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2452 CaMgO_full_tmp_2452 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2453 CaMgO_full_tmp_2453 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2454 CaMgO_full_tmp_2454 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2455 CaMgO_full_tmp_2455 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2456 CaMgO_full_tmp_2456 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2457 CaMgO_full_tmp_2457 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2458 CaMgO_full_tmp_2458 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2459 CaMgO_full_tmp_2459 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2460 CaMgO_full_tmp_2460 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2461 CaMgO_full_tmp_2461 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2462 CaMgO_full_tmp_2462 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2463 CaMgO_full_tmp_2463 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2464 CaMgO_full_tmp_2464 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2465 CaMgO_full_tmp_2465 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2466 CaMgO_full_tmp_2466 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2467 CaMgO_full_tmp_2467 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2468 CaMgO_full_tmp_2468 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2469 CaMgO_full_tmp_2469 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2470 CaMgO_full_tmp_2470 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2471 CaMgO_full_tmp_2471 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2472 CaMgO_full_tmp_2472 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2473 CaMgO_full_tmp_2473 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2474 CaMgO_full_tmp_2474 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2475 CaMgO_full_tmp_2475 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2476 CaMgO_full_tmp_2476 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2477 CaMgO_full_tmp_2477 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2478 CaMgO_full_tmp_2478 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2479 CaMgO_full_tmp_2479 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2480 CaMgO_full_tmp_2480 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2481 CaMgO_full_tmp_2481 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2482 CaMgO_full_tmp_2482 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2483 CaMgO_full_tmp_2483 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2484 CaMgO_full_tmp_2484 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2485 CaMgO_full_tmp_2485 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2486 CaMgO_full_tmp_2486 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2487 CaMgO_full_tmp_2487 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2488 CaMgO_full_tmp_2488 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2489 CaMgO_full_tmp_2489 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2490 CaMgO_full_tmp_2490 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2491 CaMgO_full_tmp_2491 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2492 CaMgO_full_tmp_2492 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2493 CaMgO_full_tmp_2493 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2494 CaMgO_full_tmp_2494 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2495 CaMgO_full_tmp_2495 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2496 CaMgO_full_tmp_2496 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2497 CaMgO_full_tmp_2497 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2498 CaMgO_full_tmp_2498 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2499 CaMgO_full_tmp_2499 &

wait
mv CaMgO_full_tmp_2400.f9 CaMgO_full_2400.f9
mv CaMgO_full_tmp_2400.gui CaMgO_full_2400.gui
mv CaMgO_full_tmp_2401.f9 CaMgO_full_2401.f9
mv CaMgO_full_tmp_2401.gui CaMgO_full_2401.gui
mv CaMgO_full_tmp_2402.f9 CaMgO_full_2402.f9
mv CaMgO_full_tmp_2402.gui CaMgO_full_2402.gui
mv CaMgO_full_tmp_2403.f9 CaMgO_full_2403.f9
mv CaMgO_full_tmp_2403.gui CaMgO_full_2403.gui
mv CaMgO_full_tmp_2404.f9 CaMgO_full_2404.f9
mv CaMgO_full_tmp_2404.gui CaMgO_full_2404.gui
mv CaMgO_full_tmp_2405.f9 CaMgO_full_2405.f9
mv CaMgO_full_tmp_2405.gui CaMgO_full_2405.gui
mv CaMgO_full_tmp_2406.f9 CaMgO_full_2406.f9
mv CaMgO_full_tmp_2406.gui CaMgO_full_2406.gui
mv CaMgO_full_tmp_2407.f9 CaMgO_full_2407.f9
mv CaMgO_full_tmp_2407.gui CaMgO_full_2407.gui
mv CaMgO_full_tmp_2408.f9 CaMgO_full_2408.f9
mv CaMgO_full_tmp_2408.gui CaMgO_full_2408.gui
mv CaMgO_full_tmp_2409.f9 CaMgO_full_2409.f9
mv CaMgO_full_tmp_2409.gui CaMgO_full_2409.gui
mv CaMgO_full_tmp_2410.f9 CaMgO_full_2410.f9
mv CaMgO_full_tmp_2410.gui CaMgO_full_2410.gui
mv CaMgO_full_tmp_2411.f9 CaMgO_full_2411.f9
mv CaMgO_full_tmp_2411.gui CaMgO_full_2411.gui
mv CaMgO_full_tmp_2412.f9 CaMgO_full_2412.f9
mv CaMgO_full_tmp_2412.gui CaMgO_full_2412.gui
mv CaMgO_full_tmp_2413.f9 CaMgO_full_2413.f9
mv CaMgO_full_tmp_2413.gui CaMgO_full_2413.gui
mv CaMgO_full_tmp_2414.f9 CaMgO_full_2414.f9
mv CaMgO_full_tmp_2414.gui CaMgO_full_2414.gui
mv CaMgO_full_tmp_2415.f9 CaMgO_full_2415.f9
mv CaMgO_full_tmp_2415.gui CaMgO_full_2415.gui
mv CaMgO_full_tmp_2416.f9 CaMgO_full_2416.f9
mv CaMgO_full_tmp_2416.gui CaMgO_full_2416.gui
mv CaMgO_full_tmp_2417.f9 CaMgO_full_2417.f9
mv CaMgO_full_tmp_2417.gui CaMgO_full_2417.gui
mv CaMgO_full_tmp_2418.f9 CaMgO_full_2418.f9
mv CaMgO_full_tmp_2418.gui CaMgO_full_2418.gui
mv CaMgO_full_tmp_2419.f9 CaMgO_full_2419.f9
mv CaMgO_full_tmp_2419.gui CaMgO_full_2419.gui
mv CaMgO_full_tmp_2420.f9 CaMgO_full_2420.f9
mv CaMgO_full_tmp_2420.gui CaMgO_full_2420.gui
mv CaMgO_full_tmp_2421.f9 CaMgO_full_2421.f9
mv CaMgO_full_tmp_2421.gui CaMgO_full_2421.gui
mv CaMgO_full_tmp_2422.f9 CaMgO_full_2422.f9
mv CaMgO_full_tmp_2422.gui CaMgO_full_2422.gui
mv CaMgO_full_tmp_2423.f9 CaMgO_full_2423.f9
mv CaMgO_full_tmp_2423.gui CaMgO_full_2423.gui
mv CaMgO_full_tmp_2424.f9 CaMgO_full_2424.f9
mv CaMgO_full_tmp_2424.gui CaMgO_full_2424.gui
mv CaMgO_full_tmp_2425.f9 CaMgO_full_2425.f9
mv CaMgO_full_tmp_2425.gui CaMgO_full_2425.gui
mv CaMgO_full_tmp_2426.f9 CaMgO_full_2426.f9
mv CaMgO_full_tmp_2426.gui CaMgO_full_2426.gui
mv CaMgO_full_tmp_2427.f9 CaMgO_full_2427.f9
mv CaMgO_full_tmp_2427.gui CaMgO_full_2427.gui
mv CaMgO_full_tmp_2428.f9 CaMgO_full_2428.f9
mv CaMgO_full_tmp_2428.gui CaMgO_full_2428.gui
mv CaMgO_full_tmp_2429.f9 CaMgO_full_2429.f9
mv CaMgO_full_tmp_2429.gui CaMgO_full_2429.gui
mv CaMgO_full_tmp_2430.f9 CaMgO_full_2430.f9
mv CaMgO_full_tmp_2430.gui CaMgO_full_2430.gui
mv CaMgO_full_tmp_2431.f9 CaMgO_full_2431.f9
mv CaMgO_full_tmp_2431.gui CaMgO_full_2431.gui
mv CaMgO_full_tmp_2432.f9 CaMgO_full_2432.f9
mv CaMgO_full_tmp_2432.gui CaMgO_full_2432.gui
mv CaMgO_full_tmp_2433.f9 CaMgO_full_2433.f9
mv CaMgO_full_tmp_2433.gui CaMgO_full_2433.gui
mv CaMgO_full_tmp_2434.f9 CaMgO_full_2434.f9
mv CaMgO_full_tmp_2434.gui CaMgO_full_2434.gui
mv CaMgO_full_tmp_2435.f9 CaMgO_full_2435.f9
mv CaMgO_full_tmp_2435.gui CaMgO_full_2435.gui
mv CaMgO_full_tmp_2436.f9 CaMgO_full_2436.f9
mv CaMgO_full_tmp_2436.gui CaMgO_full_2436.gui
mv CaMgO_full_tmp_2437.f9 CaMgO_full_2437.f9
mv CaMgO_full_tmp_2437.gui CaMgO_full_2437.gui
mv CaMgO_full_tmp_2438.f9 CaMgO_full_2438.f9
mv CaMgO_full_tmp_2438.gui CaMgO_full_2438.gui
mv CaMgO_full_tmp_2439.f9 CaMgO_full_2439.f9
mv CaMgO_full_tmp_2439.gui CaMgO_full_2439.gui
mv CaMgO_full_tmp_2440.f9 CaMgO_full_2440.f9
mv CaMgO_full_tmp_2440.gui CaMgO_full_2440.gui
mv CaMgO_full_tmp_2441.f9 CaMgO_full_2441.f9
mv CaMgO_full_tmp_2441.gui CaMgO_full_2441.gui
mv CaMgO_full_tmp_2442.f9 CaMgO_full_2442.f9
mv CaMgO_full_tmp_2442.gui CaMgO_full_2442.gui
mv CaMgO_full_tmp_2443.f9 CaMgO_full_2443.f9
mv CaMgO_full_tmp_2443.gui CaMgO_full_2443.gui
mv CaMgO_full_tmp_2444.f9 CaMgO_full_2444.f9
mv CaMgO_full_tmp_2444.gui CaMgO_full_2444.gui
mv CaMgO_full_tmp_2445.f9 CaMgO_full_2445.f9
mv CaMgO_full_tmp_2445.gui CaMgO_full_2445.gui
mv CaMgO_full_tmp_2446.f9 CaMgO_full_2446.f9
mv CaMgO_full_tmp_2446.gui CaMgO_full_2446.gui
mv CaMgO_full_tmp_2447.f9 CaMgO_full_2447.f9
mv CaMgO_full_tmp_2447.gui CaMgO_full_2447.gui
mv CaMgO_full_tmp_2448.f9 CaMgO_full_2448.f9
mv CaMgO_full_tmp_2448.gui CaMgO_full_2448.gui
mv CaMgO_full_tmp_2449.f9 CaMgO_full_2449.f9
mv CaMgO_full_tmp_2449.gui CaMgO_full_2449.gui
mv CaMgO_full_tmp_2450.f9 CaMgO_full_2450.f9
mv CaMgO_full_tmp_2450.gui CaMgO_full_2450.gui
mv CaMgO_full_tmp_2451.f9 CaMgO_full_2451.f9
mv CaMgO_full_tmp_2451.gui CaMgO_full_2451.gui
mv CaMgO_full_tmp_2452.f9 CaMgO_full_2452.f9
mv CaMgO_full_tmp_2452.gui CaMgO_full_2452.gui
mv CaMgO_full_tmp_2453.f9 CaMgO_full_2453.f9
mv CaMgO_full_tmp_2453.gui CaMgO_full_2453.gui
mv CaMgO_full_tmp_2454.f9 CaMgO_full_2454.f9
mv CaMgO_full_tmp_2454.gui CaMgO_full_2454.gui
mv CaMgO_full_tmp_2455.f9 CaMgO_full_2455.f9
mv CaMgO_full_tmp_2455.gui CaMgO_full_2455.gui
mv CaMgO_full_tmp_2456.f9 CaMgO_full_2456.f9
mv CaMgO_full_tmp_2456.gui CaMgO_full_2456.gui
mv CaMgO_full_tmp_2457.f9 CaMgO_full_2457.f9
mv CaMgO_full_tmp_2457.gui CaMgO_full_2457.gui
mv CaMgO_full_tmp_2458.f9 CaMgO_full_2458.f9
mv CaMgO_full_tmp_2458.gui CaMgO_full_2458.gui
mv CaMgO_full_tmp_2459.f9 CaMgO_full_2459.f9
mv CaMgO_full_tmp_2459.gui CaMgO_full_2459.gui
mv CaMgO_full_tmp_2460.f9 CaMgO_full_2460.f9
mv CaMgO_full_tmp_2460.gui CaMgO_full_2460.gui
mv CaMgO_full_tmp_2461.f9 CaMgO_full_2461.f9
mv CaMgO_full_tmp_2461.gui CaMgO_full_2461.gui
mv CaMgO_full_tmp_2462.f9 CaMgO_full_2462.f9
mv CaMgO_full_tmp_2462.gui CaMgO_full_2462.gui
mv CaMgO_full_tmp_2463.f9 CaMgO_full_2463.f9
mv CaMgO_full_tmp_2463.gui CaMgO_full_2463.gui
mv CaMgO_full_tmp_2464.f9 CaMgO_full_2464.f9
mv CaMgO_full_tmp_2464.gui CaMgO_full_2464.gui
mv CaMgO_full_tmp_2465.f9 CaMgO_full_2465.f9
mv CaMgO_full_tmp_2465.gui CaMgO_full_2465.gui
mv CaMgO_full_tmp_2466.f9 CaMgO_full_2466.f9
mv CaMgO_full_tmp_2466.gui CaMgO_full_2466.gui
mv CaMgO_full_tmp_2467.f9 CaMgO_full_2467.f9
mv CaMgO_full_tmp_2467.gui CaMgO_full_2467.gui
mv CaMgO_full_tmp_2468.f9 CaMgO_full_2468.f9
mv CaMgO_full_tmp_2468.gui CaMgO_full_2468.gui
mv CaMgO_full_tmp_2469.f9 CaMgO_full_2469.f9
mv CaMgO_full_tmp_2469.gui CaMgO_full_2469.gui
mv CaMgO_full_tmp_2470.f9 CaMgO_full_2470.f9
mv CaMgO_full_tmp_2470.gui CaMgO_full_2470.gui
mv CaMgO_full_tmp_2471.f9 CaMgO_full_2471.f9
mv CaMgO_full_tmp_2471.gui CaMgO_full_2471.gui
mv CaMgO_full_tmp_2472.f9 CaMgO_full_2472.f9
mv CaMgO_full_tmp_2472.gui CaMgO_full_2472.gui
mv CaMgO_full_tmp_2473.f9 CaMgO_full_2473.f9
mv CaMgO_full_tmp_2473.gui CaMgO_full_2473.gui
mv CaMgO_full_tmp_2474.f9 CaMgO_full_2474.f9
mv CaMgO_full_tmp_2474.gui CaMgO_full_2474.gui
mv CaMgO_full_tmp_2475.f9 CaMgO_full_2475.f9
mv CaMgO_full_tmp_2475.gui CaMgO_full_2475.gui
mv CaMgO_full_tmp_2476.f9 CaMgO_full_2476.f9
mv CaMgO_full_tmp_2476.gui CaMgO_full_2476.gui
mv CaMgO_full_tmp_2477.f9 CaMgO_full_2477.f9
mv CaMgO_full_tmp_2477.gui CaMgO_full_2477.gui
mv CaMgO_full_tmp_2478.f9 CaMgO_full_2478.f9
mv CaMgO_full_tmp_2478.gui CaMgO_full_2478.gui
mv CaMgO_full_tmp_2479.f9 CaMgO_full_2479.f9
mv CaMgO_full_tmp_2479.gui CaMgO_full_2479.gui
mv CaMgO_full_tmp_2480.f9 CaMgO_full_2480.f9
mv CaMgO_full_tmp_2480.gui CaMgO_full_2480.gui
mv CaMgO_full_tmp_2481.f9 CaMgO_full_2481.f9
mv CaMgO_full_tmp_2481.gui CaMgO_full_2481.gui
mv CaMgO_full_tmp_2482.f9 CaMgO_full_2482.f9
mv CaMgO_full_tmp_2482.gui CaMgO_full_2482.gui
mv CaMgO_full_tmp_2483.f9 CaMgO_full_2483.f9
mv CaMgO_full_tmp_2483.gui CaMgO_full_2483.gui
mv CaMgO_full_tmp_2484.f9 CaMgO_full_2484.f9
mv CaMgO_full_tmp_2484.gui CaMgO_full_2484.gui
mv CaMgO_full_tmp_2485.f9 CaMgO_full_2485.f9
mv CaMgO_full_tmp_2485.gui CaMgO_full_2485.gui
mv CaMgO_full_tmp_2486.f9 CaMgO_full_2486.f9
mv CaMgO_full_tmp_2486.gui CaMgO_full_2486.gui
mv CaMgO_full_tmp_2487.f9 CaMgO_full_2487.f9
mv CaMgO_full_tmp_2487.gui CaMgO_full_2487.gui
mv CaMgO_full_tmp_2488.f9 CaMgO_full_2488.f9
mv CaMgO_full_tmp_2488.gui CaMgO_full_2488.gui
mv CaMgO_full_tmp_2489.f9 CaMgO_full_2489.f9
mv CaMgO_full_tmp_2489.gui CaMgO_full_2489.gui
mv CaMgO_full_tmp_2490.f9 CaMgO_full_2490.f9
mv CaMgO_full_tmp_2490.gui CaMgO_full_2490.gui
mv CaMgO_full_tmp_2491.f9 CaMgO_full_2491.f9
mv CaMgO_full_tmp_2491.gui CaMgO_full_2491.gui
mv CaMgO_full_tmp_2492.f9 CaMgO_full_2492.f9
mv CaMgO_full_tmp_2492.gui CaMgO_full_2492.gui
mv CaMgO_full_tmp_2493.f9 CaMgO_full_2493.f9
mv CaMgO_full_tmp_2493.gui CaMgO_full_2493.gui
mv CaMgO_full_tmp_2494.f9 CaMgO_full_2494.f9
mv CaMgO_full_tmp_2494.gui CaMgO_full_2494.gui
mv CaMgO_full_tmp_2495.f9 CaMgO_full_2495.f9
mv CaMgO_full_tmp_2495.gui CaMgO_full_2495.gui
mv CaMgO_full_tmp_2496.f9 CaMgO_full_2496.f9
mv CaMgO_full_tmp_2496.gui CaMgO_full_2496.gui
mv CaMgO_full_tmp_2497.f9 CaMgO_full_2497.f9
mv CaMgO_full_tmp_2497.gui CaMgO_full_2497.gui
mv CaMgO_full_tmp_2498.f9 CaMgO_full_2498.f9
mv CaMgO_full_tmp_2498.gui CaMgO_full_2498.gui
mv CaMgO_full_tmp_2499.f9 CaMgO_full_2499.f9
mv CaMgO_full_tmp_2499.gui CaMgO_full_2499.gui
