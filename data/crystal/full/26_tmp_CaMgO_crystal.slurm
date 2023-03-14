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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2500 CaMgO_full_tmp_2500 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2501 CaMgO_full_tmp_2501 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2502 CaMgO_full_tmp_2502 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2503 CaMgO_full_tmp_2503 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2504 CaMgO_full_tmp_2504 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2505 CaMgO_full_tmp_2505 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2506 CaMgO_full_tmp_2506 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2507 CaMgO_full_tmp_2507 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2508 CaMgO_full_tmp_2508 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2509 CaMgO_full_tmp_2509 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2510 CaMgO_full_tmp_2510 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2511 CaMgO_full_tmp_2511 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2512 CaMgO_full_tmp_2512 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2513 CaMgO_full_tmp_2513 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2514 CaMgO_full_tmp_2514 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2515 CaMgO_full_tmp_2515 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2516 CaMgO_full_tmp_2516 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2517 CaMgO_full_tmp_2517 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2518 CaMgO_full_tmp_2518 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2519 CaMgO_full_tmp_2519 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2520 CaMgO_full_tmp_2520 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2521 CaMgO_full_tmp_2521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2522 CaMgO_full_tmp_2522 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2523 CaMgO_full_tmp_2523 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2524 CaMgO_full_tmp_2524 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2525 CaMgO_full_tmp_2525 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2526 CaMgO_full_tmp_2526 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2527 CaMgO_full_tmp_2527 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2528 CaMgO_full_tmp_2528 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2529 CaMgO_full_tmp_2529 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2530 CaMgO_full_tmp_2530 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2531 CaMgO_full_tmp_2531 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2532 CaMgO_full_tmp_2532 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2533 CaMgO_full_tmp_2533 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2534 CaMgO_full_tmp_2534 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2535 CaMgO_full_tmp_2535 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2536 CaMgO_full_tmp_2536 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2537 CaMgO_full_tmp_2537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2538 CaMgO_full_tmp_2538 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2539 CaMgO_full_tmp_2539 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2540 CaMgO_full_tmp_2540 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2541 CaMgO_full_tmp_2541 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2542 CaMgO_full_tmp_2542 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2543 CaMgO_full_tmp_2543 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2544 CaMgO_full_tmp_2544 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2545 CaMgO_full_tmp_2545 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2546 CaMgO_full_tmp_2546 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2547 CaMgO_full_tmp_2547 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2548 CaMgO_full_tmp_2548 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2549 CaMgO_full_tmp_2549 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2550 CaMgO_full_tmp_2550 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2551 CaMgO_full_tmp_2551 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2552 CaMgO_full_tmp_2552 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2553 CaMgO_full_tmp_2553 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2554 CaMgO_full_tmp_2554 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2555 CaMgO_full_tmp_2555 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2556 CaMgO_full_tmp_2556 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2557 CaMgO_full_tmp_2557 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2558 CaMgO_full_tmp_2558 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2559 CaMgO_full_tmp_2559 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2560 CaMgO_full_tmp_2560 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2561 CaMgO_full_tmp_2561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2562 CaMgO_full_tmp_2562 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2563 CaMgO_full_tmp_2563 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2564 CaMgO_full_tmp_2564 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2565 CaMgO_full_tmp_2565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2566 CaMgO_full_tmp_2566 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2567 CaMgO_full_tmp_2567 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2568 CaMgO_full_tmp_2568 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2569 CaMgO_full_tmp_2569 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2570 CaMgO_full_tmp_2570 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2571 CaMgO_full_tmp_2571 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2572 CaMgO_full_tmp_2572 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2573 CaMgO_full_tmp_2573 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2574 CaMgO_full_tmp_2574 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2575 CaMgO_full_tmp_2575 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2576 CaMgO_full_tmp_2576 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2577 CaMgO_full_tmp_2577 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2578 CaMgO_full_tmp_2578 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2579 CaMgO_full_tmp_2579 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2580 CaMgO_full_tmp_2580 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2581 CaMgO_full_tmp_2581 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2582 CaMgO_full_tmp_2582 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2583 CaMgO_full_tmp_2583 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2584 CaMgO_full_tmp_2584 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2585 CaMgO_full_tmp_2585 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2586 CaMgO_full_tmp_2586 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2587 CaMgO_full_tmp_2587 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2588 CaMgO_full_tmp_2588 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2589 CaMgO_full_tmp_2589 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2590 CaMgO_full_tmp_2590 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2591 CaMgO_full_tmp_2591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2592 CaMgO_full_tmp_2592 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2593 CaMgO_full_tmp_2593 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2594 CaMgO_full_tmp_2594 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2595 CaMgO_full_tmp_2595 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2596 CaMgO_full_tmp_2596 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2597 CaMgO_full_tmp_2597 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2598 CaMgO_full_tmp_2598 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2599 CaMgO_full_tmp_2599 &

wait
mv CaMgO_full_tmp_2500.f9 CaMgO_full_2500.f9
mv CaMgO_full_tmp_2500.gui CaMgO_full_2500.gui
mv CaMgO_full_tmp_2501.f9 CaMgO_full_2501.f9
mv CaMgO_full_tmp_2501.gui CaMgO_full_2501.gui
mv CaMgO_full_tmp_2502.f9 CaMgO_full_2502.f9
mv CaMgO_full_tmp_2502.gui CaMgO_full_2502.gui
mv CaMgO_full_tmp_2503.f9 CaMgO_full_2503.f9
mv CaMgO_full_tmp_2503.gui CaMgO_full_2503.gui
mv CaMgO_full_tmp_2504.f9 CaMgO_full_2504.f9
mv CaMgO_full_tmp_2504.gui CaMgO_full_2504.gui
mv CaMgO_full_tmp_2505.f9 CaMgO_full_2505.f9
mv CaMgO_full_tmp_2505.gui CaMgO_full_2505.gui
mv CaMgO_full_tmp_2506.f9 CaMgO_full_2506.f9
mv CaMgO_full_tmp_2506.gui CaMgO_full_2506.gui
mv CaMgO_full_tmp_2507.f9 CaMgO_full_2507.f9
mv CaMgO_full_tmp_2507.gui CaMgO_full_2507.gui
mv CaMgO_full_tmp_2508.f9 CaMgO_full_2508.f9
mv CaMgO_full_tmp_2508.gui CaMgO_full_2508.gui
mv CaMgO_full_tmp_2509.f9 CaMgO_full_2509.f9
mv CaMgO_full_tmp_2509.gui CaMgO_full_2509.gui
mv CaMgO_full_tmp_2510.f9 CaMgO_full_2510.f9
mv CaMgO_full_tmp_2510.gui CaMgO_full_2510.gui
mv CaMgO_full_tmp_2511.f9 CaMgO_full_2511.f9
mv CaMgO_full_tmp_2511.gui CaMgO_full_2511.gui
mv CaMgO_full_tmp_2512.f9 CaMgO_full_2512.f9
mv CaMgO_full_tmp_2512.gui CaMgO_full_2512.gui
mv CaMgO_full_tmp_2513.f9 CaMgO_full_2513.f9
mv CaMgO_full_tmp_2513.gui CaMgO_full_2513.gui
mv CaMgO_full_tmp_2514.f9 CaMgO_full_2514.f9
mv CaMgO_full_tmp_2514.gui CaMgO_full_2514.gui
mv CaMgO_full_tmp_2515.f9 CaMgO_full_2515.f9
mv CaMgO_full_tmp_2515.gui CaMgO_full_2515.gui
mv CaMgO_full_tmp_2516.f9 CaMgO_full_2516.f9
mv CaMgO_full_tmp_2516.gui CaMgO_full_2516.gui
mv CaMgO_full_tmp_2517.f9 CaMgO_full_2517.f9
mv CaMgO_full_tmp_2517.gui CaMgO_full_2517.gui
mv CaMgO_full_tmp_2518.f9 CaMgO_full_2518.f9
mv CaMgO_full_tmp_2518.gui CaMgO_full_2518.gui
mv CaMgO_full_tmp_2519.f9 CaMgO_full_2519.f9
mv CaMgO_full_tmp_2519.gui CaMgO_full_2519.gui
mv CaMgO_full_tmp_2520.f9 CaMgO_full_2520.f9
mv CaMgO_full_tmp_2520.gui CaMgO_full_2520.gui
mv CaMgO_full_tmp_2521.f9 CaMgO_full_2521.f9
mv CaMgO_full_tmp_2521.gui CaMgO_full_2521.gui
mv CaMgO_full_tmp_2522.f9 CaMgO_full_2522.f9
mv CaMgO_full_tmp_2522.gui CaMgO_full_2522.gui
mv CaMgO_full_tmp_2523.f9 CaMgO_full_2523.f9
mv CaMgO_full_tmp_2523.gui CaMgO_full_2523.gui
mv CaMgO_full_tmp_2524.f9 CaMgO_full_2524.f9
mv CaMgO_full_tmp_2524.gui CaMgO_full_2524.gui
mv CaMgO_full_tmp_2525.f9 CaMgO_full_2525.f9
mv CaMgO_full_tmp_2525.gui CaMgO_full_2525.gui
mv CaMgO_full_tmp_2526.f9 CaMgO_full_2526.f9
mv CaMgO_full_tmp_2526.gui CaMgO_full_2526.gui
mv CaMgO_full_tmp_2527.f9 CaMgO_full_2527.f9
mv CaMgO_full_tmp_2527.gui CaMgO_full_2527.gui
mv CaMgO_full_tmp_2528.f9 CaMgO_full_2528.f9
mv CaMgO_full_tmp_2528.gui CaMgO_full_2528.gui
mv CaMgO_full_tmp_2529.f9 CaMgO_full_2529.f9
mv CaMgO_full_tmp_2529.gui CaMgO_full_2529.gui
mv CaMgO_full_tmp_2530.f9 CaMgO_full_2530.f9
mv CaMgO_full_tmp_2530.gui CaMgO_full_2530.gui
mv CaMgO_full_tmp_2531.f9 CaMgO_full_2531.f9
mv CaMgO_full_tmp_2531.gui CaMgO_full_2531.gui
mv CaMgO_full_tmp_2532.f9 CaMgO_full_2532.f9
mv CaMgO_full_tmp_2532.gui CaMgO_full_2532.gui
mv CaMgO_full_tmp_2533.f9 CaMgO_full_2533.f9
mv CaMgO_full_tmp_2533.gui CaMgO_full_2533.gui
mv CaMgO_full_tmp_2534.f9 CaMgO_full_2534.f9
mv CaMgO_full_tmp_2534.gui CaMgO_full_2534.gui
mv CaMgO_full_tmp_2535.f9 CaMgO_full_2535.f9
mv CaMgO_full_tmp_2535.gui CaMgO_full_2535.gui
mv CaMgO_full_tmp_2536.f9 CaMgO_full_2536.f9
mv CaMgO_full_tmp_2536.gui CaMgO_full_2536.gui
mv CaMgO_full_tmp_2537.f9 CaMgO_full_2537.f9
mv CaMgO_full_tmp_2537.gui CaMgO_full_2537.gui
mv CaMgO_full_tmp_2538.f9 CaMgO_full_2538.f9
mv CaMgO_full_tmp_2538.gui CaMgO_full_2538.gui
mv CaMgO_full_tmp_2539.f9 CaMgO_full_2539.f9
mv CaMgO_full_tmp_2539.gui CaMgO_full_2539.gui
mv CaMgO_full_tmp_2540.f9 CaMgO_full_2540.f9
mv CaMgO_full_tmp_2540.gui CaMgO_full_2540.gui
mv CaMgO_full_tmp_2541.f9 CaMgO_full_2541.f9
mv CaMgO_full_tmp_2541.gui CaMgO_full_2541.gui
mv CaMgO_full_tmp_2542.f9 CaMgO_full_2542.f9
mv CaMgO_full_tmp_2542.gui CaMgO_full_2542.gui
mv CaMgO_full_tmp_2543.f9 CaMgO_full_2543.f9
mv CaMgO_full_tmp_2543.gui CaMgO_full_2543.gui
mv CaMgO_full_tmp_2544.f9 CaMgO_full_2544.f9
mv CaMgO_full_tmp_2544.gui CaMgO_full_2544.gui
mv CaMgO_full_tmp_2545.f9 CaMgO_full_2545.f9
mv CaMgO_full_tmp_2545.gui CaMgO_full_2545.gui
mv CaMgO_full_tmp_2546.f9 CaMgO_full_2546.f9
mv CaMgO_full_tmp_2546.gui CaMgO_full_2546.gui
mv CaMgO_full_tmp_2547.f9 CaMgO_full_2547.f9
mv CaMgO_full_tmp_2547.gui CaMgO_full_2547.gui
mv CaMgO_full_tmp_2548.f9 CaMgO_full_2548.f9
mv CaMgO_full_tmp_2548.gui CaMgO_full_2548.gui
mv CaMgO_full_tmp_2549.f9 CaMgO_full_2549.f9
mv CaMgO_full_tmp_2549.gui CaMgO_full_2549.gui
mv CaMgO_full_tmp_2550.f9 CaMgO_full_2550.f9
mv CaMgO_full_tmp_2550.gui CaMgO_full_2550.gui
mv CaMgO_full_tmp_2551.f9 CaMgO_full_2551.f9
mv CaMgO_full_tmp_2551.gui CaMgO_full_2551.gui
mv CaMgO_full_tmp_2552.f9 CaMgO_full_2552.f9
mv CaMgO_full_tmp_2552.gui CaMgO_full_2552.gui
mv CaMgO_full_tmp_2553.f9 CaMgO_full_2553.f9
mv CaMgO_full_tmp_2553.gui CaMgO_full_2553.gui
mv CaMgO_full_tmp_2554.f9 CaMgO_full_2554.f9
mv CaMgO_full_tmp_2554.gui CaMgO_full_2554.gui
mv CaMgO_full_tmp_2555.f9 CaMgO_full_2555.f9
mv CaMgO_full_tmp_2555.gui CaMgO_full_2555.gui
mv CaMgO_full_tmp_2556.f9 CaMgO_full_2556.f9
mv CaMgO_full_tmp_2556.gui CaMgO_full_2556.gui
mv CaMgO_full_tmp_2557.f9 CaMgO_full_2557.f9
mv CaMgO_full_tmp_2557.gui CaMgO_full_2557.gui
mv CaMgO_full_tmp_2558.f9 CaMgO_full_2558.f9
mv CaMgO_full_tmp_2558.gui CaMgO_full_2558.gui
mv CaMgO_full_tmp_2559.f9 CaMgO_full_2559.f9
mv CaMgO_full_tmp_2559.gui CaMgO_full_2559.gui
mv CaMgO_full_tmp_2560.f9 CaMgO_full_2560.f9
mv CaMgO_full_tmp_2560.gui CaMgO_full_2560.gui
mv CaMgO_full_tmp_2561.f9 CaMgO_full_2561.f9
mv CaMgO_full_tmp_2561.gui CaMgO_full_2561.gui
mv CaMgO_full_tmp_2562.f9 CaMgO_full_2562.f9
mv CaMgO_full_tmp_2562.gui CaMgO_full_2562.gui
mv CaMgO_full_tmp_2563.f9 CaMgO_full_2563.f9
mv CaMgO_full_tmp_2563.gui CaMgO_full_2563.gui
mv CaMgO_full_tmp_2564.f9 CaMgO_full_2564.f9
mv CaMgO_full_tmp_2564.gui CaMgO_full_2564.gui
mv CaMgO_full_tmp_2565.f9 CaMgO_full_2565.f9
mv CaMgO_full_tmp_2565.gui CaMgO_full_2565.gui
mv CaMgO_full_tmp_2566.f9 CaMgO_full_2566.f9
mv CaMgO_full_tmp_2566.gui CaMgO_full_2566.gui
mv CaMgO_full_tmp_2567.f9 CaMgO_full_2567.f9
mv CaMgO_full_tmp_2567.gui CaMgO_full_2567.gui
mv CaMgO_full_tmp_2568.f9 CaMgO_full_2568.f9
mv CaMgO_full_tmp_2568.gui CaMgO_full_2568.gui
mv CaMgO_full_tmp_2569.f9 CaMgO_full_2569.f9
mv CaMgO_full_tmp_2569.gui CaMgO_full_2569.gui
mv CaMgO_full_tmp_2570.f9 CaMgO_full_2570.f9
mv CaMgO_full_tmp_2570.gui CaMgO_full_2570.gui
mv CaMgO_full_tmp_2571.f9 CaMgO_full_2571.f9
mv CaMgO_full_tmp_2571.gui CaMgO_full_2571.gui
mv CaMgO_full_tmp_2572.f9 CaMgO_full_2572.f9
mv CaMgO_full_tmp_2572.gui CaMgO_full_2572.gui
mv CaMgO_full_tmp_2573.f9 CaMgO_full_2573.f9
mv CaMgO_full_tmp_2573.gui CaMgO_full_2573.gui
mv CaMgO_full_tmp_2574.f9 CaMgO_full_2574.f9
mv CaMgO_full_tmp_2574.gui CaMgO_full_2574.gui
mv CaMgO_full_tmp_2575.f9 CaMgO_full_2575.f9
mv CaMgO_full_tmp_2575.gui CaMgO_full_2575.gui
mv CaMgO_full_tmp_2576.f9 CaMgO_full_2576.f9
mv CaMgO_full_tmp_2576.gui CaMgO_full_2576.gui
mv CaMgO_full_tmp_2577.f9 CaMgO_full_2577.f9
mv CaMgO_full_tmp_2577.gui CaMgO_full_2577.gui
mv CaMgO_full_tmp_2578.f9 CaMgO_full_2578.f9
mv CaMgO_full_tmp_2578.gui CaMgO_full_2578.gui
mv CaMgO_full_tmp_2579.f9 CaMgO_full_2579.f9
mv CaMgO_full_tmp_2579.gui CaMgO_full_2579.gui
mv CaMgO_full_tmp_2580.f9 CaMgO_full_2580.f9
mv CaMgO_full_tmp_2580.gui CaMgO_full_2580.gui
mv CaMgO_full_tmp_2581.f9 CaMgO_full_2581.f9
mv CaMgO_full_tmp_2581.gui CaMgO_full_2581.gui
mv CaMgO_full_tmp_2582.f9 CaMgO_full_2582.f9
mv CaMgO_full_tmp_2582.gui CaMgO_full_2582.gui
mv CaMgO_full_tmp_2583.f9 CaMgO_full_2583.f9
mv CaMgO_full_tmp_2583.gui CaMgO_full_2583.gui
mv CaMgO_full_tmp_2584.f9 CaMgO_full_2584.f9
mv CaMgO_full_tmp_2584.gui CaMgO_full_2584.gui
mv CaMgO_full_tmp_2585.f9 CaMgO_full_2585.f9
mv CaMgO_full_tmp_2585.gui CaMgO_full_2585.gui
mv CaMgO_full_tmp_2586.f9 CaMgO_full_2586.f9
mv CaMgO_full_tmp_2586.gui CaMgO_full_2586.gui
mv CaMgO_full_tmp_2587.f9 CaMgO_full_2587.f9
mv CaMgO_full_tmp_2587.gui CaMgO_full_2587.gui
mv CaMgO_full_tmp_2588.f9 CaMgO_full_2588.f9
mv CaMgO_full_tmp_2588.gui CaMgO_full_2588.gui
mv CaMgO_full_tmp_2589.f9 CaMgO_full_2589.f9
mv CaMgO_full_tmp_2589.gui CaMgO_full_2589.gui
mv CaMgO_full_tmp_2590.f9 CaMgO_full_2590.f9
mv CaMgO_full_tmp_2590.gui CaMgO_full_2590.gui
mv CaMgO_full_tmp_2591.f9 CaMgO_full_2591.f9
mv CaMgO_full_tmp_2591.gui CaMgO_full_2591.gui
mv CaMgO_full_tmp_2592.f9 CaMgO_full_2592.f9
mv CaMgO_full_tmp_2592.gui CaMgO_full_2592.gui
mv CaMgO_full_tmp_2593.f9 CaMgO_full_2593.f9
mv CaMgO_full_tmp_2593.gui CaMgO_full_2593.gui
mv CaMgO_full_tmp_2594.f9 CaMgO_full_2594.f9
mv CaMgO_full_tmp_2594.gui CaMgO_full_2594.gui
mv CaMgO_full_tmp_2595.f9 CaMgO_full_2595.f9
mv CaMgO_full_tmp_2595.gui CaMgO_full_2595.gui
mv CaMgO_full_tmp_2596.f9 CaMgO_full_2596.f9
mv CaMgO_full_tmp_2596.gui CaMgO_full_2596.gui
mv CaMgO_full_tmp_2597.f9 CaMgO_full_2597.f9
mv CaMgO_full_tmp_2597.gui CaMgO_full_2597.gui
mv CaMgO_full_tmp_2598.f9 CaMgO_full_2598.f9
mv CaMgO_full_tmp_2598.gui CaMgO_full_2598.gui
mv CaMgO_full_tmp_2599.f9 CaMgO_full_2599.f9
mv CaMgO_full_tmp_2599.gui CaMgO_full_2599.gui
