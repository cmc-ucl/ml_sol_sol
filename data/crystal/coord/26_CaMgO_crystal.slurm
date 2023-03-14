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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2501 CaMgO_coord_2501
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2501


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2502 CaMgO_coord_2502
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2502


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2503 CaMgO_coord_2503
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2503


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2504 CaMgO_coord_2504
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2504


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2505 CaMgO_coord_2505
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2505


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2506 CaMgO_coord_2506
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2506


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2507 CaMgO_coord_2507
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2507


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2508 CaMgO_coord_2508
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2508


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2509 CaMgO_coord_2509
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2509


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2510 CaMgO_coord_2510
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2510


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2511 CaMgO_coord_2511
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2511


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2512 CaMgO_coord_2512
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2512


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2513 CaMgO_coord_2513
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2513


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2514 CaMgO_coord_2514
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2514


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2515 CaMgO_coord_2515
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2515


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2516 CaMgO_coord_2516
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2516


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2517 CaMgO_coord_2517
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2517


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2518 CaMgO_coord_2518
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2518


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2519 CaMgO_coord_2519
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2519


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2520 CaMgO_coord_2520
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2520


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2521 CaMgO_coord_2521
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2521


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2522 CaMgO_coord_2522
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2522


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2523 CaMgO_coord_2523
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2523


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2524 CaMgO_coord_2524
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2524


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2525 CaMgO_coord_2525
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2525


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2526 CaMgO_coord_2526
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2526


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2527 CaMgO_coord_2527
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2527


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2528 CaMgO_coord_2528
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2528


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2529 CaMgO_coord_2529
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2529


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2530 CaMgO_coord_2530
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2530


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2531 CaMgO_coord_2531
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2531


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2532 CaMgO_coord_2532
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2532


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2533 CaMgO_coord_2533
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2533


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2534 CaMgO_coord_2534
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2534


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2535 CaMgO_coord_2535
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2535


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2536 CaMgO_coord_2536
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2536


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2537 CaMgO_coord_2537
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2537


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2538 CaMgO_coord_2538
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2538


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2539 CaMgO_coord_2539
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2539


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2540 CaMgO_coord_2540
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2540


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2541 CaMgO_coord_2541
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2541


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2542 CaMgO_coord_2542
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2542


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2543 CaMgO_coord_2543
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2543


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2544 CaMgO_coord_2544
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2544


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2545 CaMgO_coord_2545
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2545


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2546 CaMgO_coord_2546
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2546


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2547 CaMgO_coord_2547
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2547


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2548 CaMgO_coord_2548
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2548


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2549 CaMgO_coord_2549
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2549


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2550 CaMgO_coord_2550
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2550


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2551 CaMgO_coord_2551
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2551


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2552 CaMgO_coord_2552
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2552


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2553 CaMgO_coord_2553
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2553


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2554 CaMgO_coord_2554
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2554


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2555 CaMgO_coord_2555
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2555


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2556 CaMgO_coord_2556
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2556


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2557 CaMgO_coord_2557
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2557


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2558 CaMgO_coord_2558
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2558


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2559 CaMgO_coord_2559
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2559


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2560 CaMgO_coord_2560
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2560


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2561 CaMgO_coord_2561
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2561


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2562 CaMgO_coord_2562
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2562


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2563 CaMgO_coord_2563
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2563


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2564 CaMgO_coord_2564
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2564


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2565 CaMgO_coord_2565
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2565


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2566 CaMgO_coord_2566
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2566


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2567 CaMgO_coord_2567
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2567


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2568 CaMgO_coord_2568
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2568


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2569 CaMgO_coord_2569
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2569


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2570 CaMgO_coord_2570
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2570


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2571 CaMgO_coord_2571
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2571


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2572 CaMgO_coord_2572
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2572


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2573 CaMgO_coord_2573
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2573


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2574 CaMgO_coord_2574
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2574


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2575 CaMgO_coord_2575
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2575


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2576 CaMgO_coord_2576
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2576


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2577 CaMgO_coord_2577
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2577


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2578 CaMgO_coord_2578
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2578


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2579 CaMgO_coord_2579
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2579


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2580 CaMgO_coord_2580
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2580


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2581 CaMgO_coord_2581
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2581


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2582 CaMgO_coord_2582
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2582


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2583 CaMgO_coord_2583
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2583


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2584 CaMgO_coord_2584
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2584


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2585 CaMgO_coord_2585
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2585


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2586 CaMgO_coord_2586
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2586


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2587 CaMgO_coord_2587
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2587


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2588 CaMgO_coord_2588
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2588


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2589 CaMgO_coord_2589
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2589


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2590 CaMgO_coord_2590
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2590


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2591 CaMgO_coord_2591
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2591


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2592 CaMgO_coord_2592
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2592


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2593 CaMgO_coord_2593
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2593


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2594 CaMgO_coord_2594
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2594


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2595 CaMgO_coord_2595
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2595


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2596 CaMgO_coord_2596
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2596


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2597 CaMgO_coord_2597
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2597


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2598 CaMgO_coord_2598
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2598


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2599 CaMgO_coord_2599
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2599


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2600 CaMgO_coord_2600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2600


