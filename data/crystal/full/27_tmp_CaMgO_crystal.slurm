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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2600 CaMgO_full_tmp_2600 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2601 CaMgO_full_tmp_2601 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2602 CaMgO_full_tmp_2602 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2603 CaMgO_full_tmp_2603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2604 CaMgO_full_tmp_2604 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2605 CaMgO_full_tmp_2605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2606 CaMgO_full_tmp_2606 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2607 CaMgO_full_tmp_2607 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2608 CaMgO_full_tmp_2608 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2609 CaMgO_full_tmp_2609 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2610 CaMgO_full_tmp_2610 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2611 CaMgO_full_tmp_2611 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2612 CaMgO_full_tmp_2612 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2613 CaMgO_full_tmp_2613 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2614 CaMgO_full_tmp_2614 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2615 CaMgO_full_tmp_2615 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2616 CaMgO_full_tmp_2616 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2617 CaMgO_full_tmp_2617 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2618 CaMgO_full_tmp_2618 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2619 CaMgO_full_tmp_2619 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2620 CaMgO_full_tmp_2620 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2621 CaMgO_full_tmp_2621 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2622 CaMgO_full_tmp_2622 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2623 CaMgO_full_tmp_2623 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2624 CaMgO_full_tmp_2624 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2625 CaMgO_full_tmp_2625 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2626 CaMgO_full_tmp_2626 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2627 CaMgO_full_tmp_2627 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2628 CaMgO_full_tmp_2628 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2629 CaMgO_full_tmp_2629 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2630 CaMgO_full_tmp_2630 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2631 CaMgO_full_tmp_2631 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2632 CaMgO_full_tmp_2632 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2633 CaMgO_full_tmp_2633 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2634 CaMgO_full_tmp_2634 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2635 CaMgO_full_tmp_2635 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2636 CaMgO_full_tmp_2636 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2637 CaMgO_full_tmp_2637 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2638 CaMgO_full_tmp_2638 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2639 CaMgO_full_tmp_2639 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2640 CaMgO_full_tmp_2640 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2641 CaMgO_full_tmp_2641 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2642 CaMgO_full_tmp_2642 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2643 CaMgO_full_tmp_2643 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2644 CaMgO_full_tmp_2644 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2645 CaMgO_full_tmp_2645 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2646 CaMgO_full_tmp_2646 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2647 CaMgO_full_tmp_2647 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2648 CaMgO_full_tmp_2648 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2649 CaMgO_full_tmp_2649 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2650 CaMgO_full_tmp_2650 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2651 CaMgO_full_tmp_2651 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2652 CaMgO_full_tmp_2652 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2653 CaMgO_full_tmp_2653 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2654 CaMgO_full_tmp_2654 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2655 CaMgO_full_tmp_2655 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2656 CaMgO_full_tmp_2656 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2657 CaMgO_full_tmp_2657 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2658 CaMgO_full_tmp_2658 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2659 CaMgO_full_tmp_2659 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2660 CaMgO_full_tmp_2660 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2661 CaMgO_full_tmp_2661 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2662 CaMgO_full_tmp_2662 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2663 CaMgO_full_tmp_2663 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2664 CaMgO_full_tmp_2664 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2665 CaMgO_full_tmp_2665 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2666 CaMgO_full_tmp_2666 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2667 CaMgO_full_tmp_2667 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2668 CaMgO_full_tmp_2668 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2669 CaMgO_full_tmp_2669 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2670 CaMgO_full_tmp_2670 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2671 CaMgO_full_tmp_2671 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2672 CaMgO_full_tmp_2672 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2673 CaMgO_full_tmp_2673 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2674 CaMgO_full_tmp_2674 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2675 CaMgO_full_tmp_2675 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2676 CaMgO_full_tmp_2676 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2677 CaMgO_full_tmp_2677 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2678 CaMgO_full_tmp_2678 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2679 CaMgO_full_tmp_2679 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2680 CaMgO_full_tmp_2680 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2681 CaMgO_full_tmp_2681 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2682 CaMgO_full_tmp_2682 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2683 CaMgO_full_tmp_2683 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2684 CaMgO_full_tmp_2684 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2685 CaMgO_full_tmp_2685 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2686 CaMgO_full_tmp_2686 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2687 CaMgO_full_tmp_2687 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2688 CaMgO_full_tmp_2688 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2689 CaMgO_full_tmp_2689 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2690 CaMgO_full_tmp_2690 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2691 CaMgO_full_tmp_2691 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2692 CaMgO_full_tmp_2692 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2693 CaMgO_full_tmp_2693 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2694 CaMgO_full_tmp_2694 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2695 CaMgO_full_tmp_2695 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2696 CaMgO_full_tmp_2696 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2697 CaMgO_full_tmp_2697 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2698 CaMgO_full_tmp_2698 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2699 CaMgO_full_tmp_2699 &

wait
mv CaMgO_full_tmp_2600.f9 CaMgO_full_2600.f9
mv CaMgO_full_tmp_2600.gui CaMgO_full_2600.gui
mv CaMgO_full_tmp_2601.f9 CaMgO_full_2601.f9
mv CaMgO_full_tmp_2601.gui CaMgO_full_2601.gui
mv CaMgO_full_tmp_2602.f9 CaMgO_full_2602.f9
mv CaMgO_full_tmp_2602.gui CaMgO_full_2602.gui
mv CaMgO_full_tmp_2603.f9 CaMgO_full_2603.f9
mv CaMgO_full_tmp_2603.gui CaMgO_full_2603.gui
mv CaMgO_full_tmp_2604.f9 CaMgO_full_2604.f9
mv CaMgO_full_tmp_2604.gui CaMgO_full_2604.gui
mv CaMgO_full_tmp_2605.f9 CaMgO_full_2605.f9
mv CaMgO_full_tmp_2605.gui CaMgO_full_2605.gui
mv CaMgO_full_tmp_2606.f9 CaMgO_full_2606.f9
mv CaMgO_full_tmp_2606.gui CaMgO_full_2606.gui
mv CaMgO_full_tmp_2607.f9 CaMgO_full_2607.f9
mv CaMgO_full_tmp_2607.gui CaMgO_full_2607.gui
mv CaMgO_full_tmp_2608.f9 CaMgO_full_2608.f9
mv CaMgO_full_tmp_2608.gui CaMgO_full_2608.gui
mv CaMgO_full_tmp_2609.f9 CaMgO_full_2609.f9
mv CaMgO_full_tmp_2609.gui CaMgO_full_2609.gui
mv CaMgO_full_tmp_2610.f9 CaMgO_full_2610.f9
mv CaMgO_full_tmp_2610.gui CaMgO_full_2610.gui
mv CaMgO_full_tmp_2611.f9 CaMgO_full_2611.f9
mv CaMgO_full_tmp_2611.gui CaMgO_full_2611.gui
mv CaMgO_full_tmp_2612.f9 CaMgO_full_2612.f9
mv CaMgO_full_tmp_2612.gui CaMgO_full_2612.gui
mv CaMgO_full_tmp_2613.f9 CaMgO_full_2613.f9
mv CaMgO_full_tmp_2613.gui CaMgO_full_2613.gui
mv CaMgO_full_tmp_2614.f9 CaMgO_full_2614.f9
mv CaMgO_full_tmp_2614.gui CaMgO_full_2614.gui
mv CaMgO_full_tmp_2615.f9 CaMgO_full_2615.f9
mv CaMgO_full_tmp_2615.gui CaMgO_full_2615.gui
mv CaMgO_full_tmp_2616.f9 CaMgO_full_2616.f9
mv CaMgO_full_tmp_2616.gui CaMgO_full_2616.gui
mv CaMgO_full_tmp_2617.f9 CaMgO_full_2617.f9
mv CaMgO_full_tmp_2617.gui CaMgO_full_2617.gui
mv CaMgO_full_tmp_2618.f9 CaMgO_full_2618.f9
mv CaMgO_full_tmp_2618.gui CaMgO_full_2618.gui
mv CaMgO_full_tmp_2619.f9 CaMgO_full_2619.f9
mv CaMgO_full_tmp_2619.gui CaMgO_full_2619.gui
mv CaMgO_full_tmp_2620.f9 CaMgO_full_2620.f9
mv CaMgO_full_tmp_2620.gui CaMgO_full_2620.gui
mv CaMgO_full_tmp_2621.f9 CaMgO_full_2621.f9
mv CaMgO_full_tmp_2621.gui CaMgO_full_2621.gui
mv CaMgO_full_tmp_2622.f9 CaMgO_full_2622.f9
mv CaMgO_full_tmp_2622.gui CaMgO_full_2622.gui
mv CaMgO_full_tmp_2623.f9 CaMgO_full_2623.f9
mv CaMgO_full_tmp_2623.gui CaMgO_full_2623.gui
mv CaMgO_full_tmp_2624.f9 CaMgO_full_2624.f9
mv CaMgO_full_tmp_2624.gui CaMgO_full_2624.gui
mv CaMgO_full_tmp_2625.f9 CaMgO_full_2625.f9
mv CaMgO_full_tmp_2625.gui CaMgO_full_2625.gui
mv CaMgO_full_tmp_2626.f9 CaMgO_full_2626.f9
mv CaMgO_full_tmp_2626.gui CaMgO_full_2626.gui
mv CaMgO_full_tmp_2627.f9 CaMgO_full_2627.f9
mv CaMgO_full_tmp_2627.gui CaMgO_full_2627.gui
mv CaMgO_full_tmp_2628.f9 CaMgO_full_2628.f9
mv CaMgO_full_tmp_2628.gui CaMgO_full_2628.gui
mv CaMgO_full_tmp_2629.f9 CaMgO_full_2629.f9
mv CaMgO_full_tmp_2629.gui CaMgO_full_2629.gui
mv CaMgO_full_tmp_2630.f9 CaMgO_full_2630.f9
mv CaMgO_full_tmp_2630.gui CaMgO_full_2630.gui
mv CaMgO_full_tmp_2631.f9 CaMgO_full_2631.f9
mv CaMgO_full_tmp_2631.gui CaMgO_full_2631.gui
mv CaMgO_full_tmp_2632.f9 CaMgO_full_2632.f9
mv CaMgO_full_tmp_2632.gui CaMgO_full_2632.gui
mv CaMgO_full_tmp_2633.f9 CaMgO_full_2633.f9
mv CaMgO_full_tmp_2633.gui CaMgO_full_2633.gui
mv CaMgO_full_tmp_2634.f9 CaMgO_full_2634.f9
mv CaMgO_full_tmp_2634.gui CaMgO_full_2634.gui
mv CaMgO_full_tmp_2635.f9 CaMgO_full_2635.f9
mv CaMgO_full_tmp_2635.gui CaMgO_full_2635.gui
mv CaMgO_full_tmp_2636.f9 CaMgO_full_2636.f9
mv CaMgO_full_tmp_2636.gui CaMgO_full_2636.gui
mv CaMgO_full_tmp_2637.f9 CaMgO_full_2637.f9
mv CaMgO_full_tmp_2637.gui CaMgO_full_2637.gui
mv CaMgO_full_tmp_2638.f9 CaMgO_full_2638.f9
mv CaMgO_full_tmp_2638.gui CaMgO_full_2638.gui
mv CaMgO_full_tmp_2639.f9 CaMgO_full_2639.f9
mv CaMgO_full_tmp_2639.gui CaMgO_full_2639.gui
mv CaMgO_full_tmp_2640.f9 CaMgO_full_2640.f9
mv CaMgO_full_tmp_2640.gui CaMgO_full_2640.gui
mv CaMgO_full_tmp_2641.f9 CaMgO_full_2641.f9
mv CaMgO_full_tmp_2641.gui CaMgO_full_2641.gui
mv CaMgO_full_tmp_2642.f9 CaMgO_full_2642.f9
mv CaMgO_full_tmp_2642.gui CaMgO_full_2642.gui
mv CaMgO_full_tmp_2643.f9 CaMgO_full_2643.f9
mv CaMgO_full_tmp_2643.gui CaMgO_full_2643.gui
mv CaMgO_full_tmp_2644.f9 CaMgO_full_2644.f9
mv CaMgO_full_tmp_2644.gui CaMgO_full_2644.gui
mv CaMgO_full_tmp_2645.f9 CaMgO_full_2645.f9
mv CaMgO_full_tmp_2645.gui CaMgO_full_2645.gui
mv CaMgO_full_tmp_2646.f9 CaMgO_full_2646.f9
mv CaMgO_full_tmp_2646.gui CaMgO_full_2646.gui
mv CaMgO_full_tmp_2647.f9 CaMgO_full_2647.f9
mv CaMgO_full_tmp_2647.gui CaMgO_full_2647.gui
mv CaMgO_full_tmp_2648.f9 CaMgO_full_2648.f9
mv CaMgO_full_tmp_2648.gui CaMgO_full_2648.gui
mv CaMgO_full_tmp_2649.f9 CaMgO_full_2649.f9
mv CaMgO_full_tmp_2649.gui CaMgO_full_2649.gui
mv CaMgO_full_tmp_2650.f9 CaMgO_full_2650.f9
mv CaMgO_full_tmp_2650.gui CaMgO_full_2650.gui
mv CaMgO_full_tmp_2651.f9 CaMgO_full_2651.f9
mv CaMgO_full_tmp_2651.gui CaMgO_full_2651.gui
mv CaMgO_full_tmp_2652.f9 CaMgO_full_2652.f9
mv CaMgO_full_tmp_2652.gui CaMgO_full_2652.gui
mv CaMgO_full_tmp_2653.f9 CaMgO_full_2653.f9
mv CaMgO_full_tmp_2653.gui CaMgO_full_2653.gui
mv CaMgO_full_tmp_2654.f9 CaMgO_full_2654.f9
mv CaMgO_full_tmp_2654.gui CaMgO_full_2654.gui
mv CaMgO_full_tmp_2655.f9 CaMgO_full_2655.f9
mv CaMgO_full_tmp_2655.gui CaMgO_full_2655.gui
mv CaMgO_full_tmp_2656.f9 CaMgO_full_2656.f9
mv CaMgO_full_tmp_2656.gui CaMgO_full_2656.gui
mv CaMgO_full_tmp_2657.f9 CaMgO_full_2657.f9
mv CaMgO_full_tmp_2657.gui CaMgO_full_2657.gui
mv CaMgO_full_tmp_2658.f9 CaMgO_full_2658.f9
mv CaMgO_full_tmp_2658.gui CaMgO_full_2658.gui
mv CaMgO_full_tmp_2659.f9 CaMgO_full_2659.f9
mv CaMgO_full_tmp_2659.gui CaMgO_full_2659.gui
mv CaMgO_full_tmp_2660.f9 CaMgO_full_2660.f9
mv CaMgO_full_tmp_2660.gui CaMgO_full_2660.gui
mv CaMgO_full_tmp_2661.f9 CaMgO_full_2661.f9
mv CaMgO_full_tmp_2661.gui CaMgO_full_2661.gui
mv CaMgO_full_tmp_2662.f9 CaMgO_full_2662.f9
mv CaMgO_full_tmp_2662.gui CaMgO_full_2662.gui
mv CaMgO_full_tmp_2663.f9 CaMgO_full_2663.f9
mv CaMgO_full_tmp_2663.gui CaMgO_full_2663.gui
mv CaMgO_full_tmp_2664.f9 CaMgO_full_2664.f9
mv CaMgO_full_tmp_2664.gui CaMgO_full_2664.gui
mv CaMgO_full_tmp_2665.f9 CaMgO_full_2665.f9
mv CaMgO_full_tmp_2665.gui CaMgO_full_2665.gui
mv CaMgO_full_tmp_2666.f9 CaMgO_full_2666.f9
mv CaMgO_full_tmp_2666.gui CaMgO_full_2666.gui
mv CaMgO_full_tmp_2667.f9 CaMgO_full_2667.f9
mv CaMgO_full_tmp_2667.gui CaMgO_full_2667.gui
mv CaMgO_full_tmp_2668.f9 CaMgO_full_2668.f9
mv CaMgO_full_tmp_2668.gui CaMgO_full_2668.gui
mv CaMgO_full_tmp_2669.f9 CaMgO_full_2669.f9
mv CaMgO_full_tmp_2669.gui CaMgO_full_2669.gui
mv CaMgO_full_tmp_2670.f9 CaMgO_full_2670.f9
mv CaMgO_full_tmp_2670.gui CaMgO_full_2670.gui
mv CaMgO_full_tmp_2671.f9 CaMgO_full_2671.f9
mv CaMgO_full_tmp_2671.gui CaMgO_full_2671.gui
mv CaMgO_full_tmp_2672.f9 CaMgO_full_2672.f9
mv CaMgO_full_tmp_2672.gui CaMgO_full_2672.gui
mv CaMgO_full_tmp_2673.f9 CaMgO_full_2673.f9
mv CaMgO_full_tmp_2673.gui CaMgO_full_2673.gui
mv CaMgO_full_tmp_2674.f9 CaMgO_full_2674.f9
mv CaMgO_full_tmp_2674.gui CaMgO_full_2674.gui
mv CaMgO_full_tmp_2675.f9 CaMgO_full_2675.f9
mv CaMgO_full_tmp_2675.gui CaMgO_full_2675.gui
mv CaMgO_full_tmp_2676.f9 CaMgO_full_2676.f9
mv CaMgO_full_tmp_2676.gui CaMgO_full_2676.gui
mv CaMgO_full_tmp_2677.f9 CaMgO_full_2677.f9
mv CaMgO_full_tmp_2677.gui CaMgO_full_2677.gui
mv CaMgO_full_tmp_2678.f9 CaMgO_full_2678.f9
mv CaMgO_full_tmp_2678.gui CaMgO_full_2678.gui
mv CaMgO_full_tmp_2679.f9 CaMgO_full_2679.f9
mv CaMgO_full_tmp_2679.gui CaMgO_full_2679.gui
mv CaMgO_full_tmp_2680.f9 CaMgO_full_2680.f9
mv CaMgO_full_tmp_2680.gui CaMgO_full_2680.gui
mv CaMgO_full_tmp_2681.f9 CaMgO_full_2681.f9
mv CaMgO_full_tmp_2681.gui CaMgO_full_2681.gui
mv CaMgO_full_tmp_2682.f9 CaMgO_full_2682.f9
mv CaMgO_full_tmp_2682.gui CaMgO_full_2682.gui
mv CaMgO_full_tmp_2683.f9 CaMgO_full_2683.f9
mv CaMgO_full_tmp_2683.gui CaMgO_full_2683.gui
mv CaMgO_full_tmp_2684.f9 CaMgO_full_2684.f9
mv CaMgO_full_tmp_2684.gui CaMgO_full_2684.gui
mv CaMgO_full_tmp_2685.f9 CaMgO_full_2685.f9
mv CaMgO_full_tmp_2685.gui CaMgO_full_2685.gui
mv CaMgO_full_tmp_2686.f9 CaMgO_full_2686.f9
mv CaMgO_full_tmp_2686.gui CaMgO_full_2686.gui
mv CaMgO_full_tmp_2687.f9 CaMgO_full_2687.f9
mv CaMgO_full_tmp_2687.gui CaMgO_full_2687.gui
mv CaMgO_full_tmp_2688.f9 CaMgO_full_2688.f9
mv CaMgO_full_tmp_2688.gui CaMgO_full_2688.gui
mv CaMgO_full_tmp_2689.f9 CaMgO_full_2689.f9
mv CaMgO_full_tmp_2689.gui CaMgO_full_2689.gui
mv CaMgO_full_tmp_2690.f9 CaMgO_full_2690.f9
mv CaMgO_full_tmp_2690.gui CaMgO_full_2690.gui
mv CaMgO_full_tmp_2691.f9 CaMgO_full_2691.f9
mv CaMgO_full_tmp_2691.gui CaMgO_full_2691.gui
mv CaMgO_full_tmp_2692.f9 CaMgO_full_2692.f9
mv CaMgO_full_tmp_2692.gui CaMgO_full_2692.gui
mv CaMgO_full_tmp_2693.f9 CaMgO_full_2693.f9
mv CaMgO_full_tmp_2693.gui CaMgO_full_2693.gui
mv CaMgO_full_tmp_2694.f9 CaMgO_full_2694.f9
mv CaMgO_full_tmp_2694.gui CaMgO_full_2694.gui
mv CaMgO_full_tmp_2695.f9 CaMgO_full_2695.f9
mv CaMgO_full_tmp_2695.gui CaMgO_full_2695.gui
mv CaMgO_full_tmp_2696.f9 CaMgO_full_2696.f9
mv CaMgO_full_tmp_2696.gui CaMgO_full_2696.gui
mv CaMgO_full_tmp_2697.f9 CaMgO_full_2697.f9
mv CaMgO_full_tmp_2697.gui CaMgO_full_2697.gui
mv CaMgO_full_tmp_2698.f9 CaMgO_full_2698.f9
mv CaMgO_full_tmp_2698.gui CaMgO_full_2698.gui
mv CaMgO_full_tmp_2699.f9 CaMgO_full_2699.f9
mv CaMgO_full_tmp_2699.gui CaMgO_full_2699.gui
