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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2601 CaMgO_coord_2601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2602 CaMgO_coord_2602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2603 CaMgO_coord_2603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2604 CaMgO_coord_2604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2605 CaMgO_coord_2605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2606 CaMgO_coord_2606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2607 CaMgO_coord_2607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2608 CaMgO_coord_2608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2609 CaMgO_coord_2609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2610 CaMgO_coord_2610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2611 CaMgO_coord_2611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2612 CaMgO_coord_2612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2613 CaMgO_coord_2613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2614 CaMgO_coord_2614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2615 CaMgO_coord_2615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2616 CaMgO_coord_2616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2617 CaMgO_coord_2617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2618 CaMgO_coord_2618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2619 CaMgO_coord_2619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2620 CaMgO_coord_2620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2621 CaMgO_coord_2621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2622 CaMgO_coord_2622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2623 CaMgO_coord_2623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2624 CaMgO_coord_2624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2625 CaMgO_coord_2625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2626 CaMgO_coord_2626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2627 CaMgO_coord_2627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2628 CaMgO_coord_2628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2629 CaMgO_coord_2629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2630 CaMgO_coord_2630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2631 CaMgO_coord_2631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2632 CaMgO_coord_2632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2633 CaMgO_coord_2633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2634 CaMgO_coord_2634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2635 CaMgO_coord_2635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2636 CaMgO_coord_2636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2637 CaMgO_coord_2637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2638 CaMgO_coord_2638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2639 CaMgO_coord_2639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2640 CaMgO_coord_2640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2641 CaMgO_coord_2641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2642 CaMgO_coord_2642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2643 CaMgO_coord_2643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2644 CaMgO_coord_2644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2645 CaMgO_coord_2645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2646 CaMgO_coord_2646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2647 CaMgO_coord_2647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2648 CaMgO_coord_2648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2649 CaMgO_coord_2649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2650 CaMgO_coord_2650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2651 CaMgO_coord_2651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2652 CaMgO_coord_2652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2653 CaMgO_coord_2653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2654 CaMgO_coord_2654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2655 CaMgO_coord_2655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2656 CaMgO_coord_2656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2657 CaMgO_coord_2657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2658 CaMgO_coord_2658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2659 CaMgO_coord_2659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2660 CaMgO_coord_2660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2661 CaMgO_coord_2661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2662 CaMgO_coord_2662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2663 CaMgO_coord_2663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2664 CaMgO_coord_2664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2665 CaMgO_coord_2665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2666 CaMgO_coord_2666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2667 CaMgO_coord_2667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2668 CaMgO_coord_2668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2669 CaMgO_coord_2669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2670 CaMgO_coord_2670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2671 CaMgO_coord_2671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2672 CaMgO_coord_2672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2673 CaMgO_coord_2673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2674 CaMgO_coord_2674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2675 CaMgO_coord_2675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2676 CaMgO_coord_2676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2677 CaMgO_coord_2677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2678 CaMgO_coord_2678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2679 CaMgO_coord_2679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2680 CaMgO_coord_2680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2681 CaMgO_coord_2681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2682 CaMgO_coord_2682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2683 CaMgO_coord_2683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2684 CaMgO_coord_2684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2685 CaMgO_coord_2685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2686 CaMgO_coord_2686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2687 CaMgO_coord_2687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2688 CaMgO_coord_2688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2689 CaMgO_coord_2689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2690 CaMgO_coord_2690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2691 CaMgO_coord_2691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2692 CaMgO_coord_2692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2693 CaMgO_coord_2693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2694 CaMgO_coord_2694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2695 CaMgO_coord_2695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2696 CaMgO_coord_2696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2697 CaMgO_coord_2697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2698 CaMgO_coord_2698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2699 CaMgO_coord_2699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2699


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2700 CaMgO_coord_2700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2700


