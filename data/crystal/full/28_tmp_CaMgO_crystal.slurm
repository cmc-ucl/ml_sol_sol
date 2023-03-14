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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2700 CaMgO_full_tmp_2700 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2701 CaMgO_full_tmp_2701 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2702 CaMgO_full_tmp_2702 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2703 CaMgO_full_tmp_2703 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2704 CaMgO_full_tmp_2704 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2705 CaMgO_full_tmp_2705 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2706 CaMgO_full_tmp_2706 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2707 CaMgO_full_tmp_2707 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2708 CaMgO_full_tmp_2708 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2709 CaMgO_full_tmp_2709 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2710 CaMgO_full_tmp_2710 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2711 CaMgO_full_tmp_2711 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2712 CaMgO_full_tmp_2712 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2713 CaMgO_full_tmp_2713 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2714 CaMgO_full_tmp_2714 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2715 CaMgO_full_tmp_2715 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2716 CaMgO_full_tmp_2716 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2717 CaMgO_full_tmp_2717 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2718 CaMgO_full_tmp_2718 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2719 CaMgO_full_tmp_2719 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2720 CaMgO_full_tmp_2720 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2721 CaMgO_full_tmp_2721 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2722 CaMgO_full_tmp_2722 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2723 CaMgO_full_tmp_2723 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2724 CaMgO_full_tmp_2724 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2725 CaMgO_full_tmp_2725 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2726 CaMgO_full_tmp_2726 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2727 CaMgO_full_tmp_2727 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2728 CaMgO_full_tmp_2728 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2729 CaMgO_full_tmp_2729 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2730 CaMgO_full_tmp_2730 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2731 CaMgO_full_tmp_2731 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2732 CaMgO_full_tmp_2732 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2733 CaMgO_full_tmp_2733 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2734 CaMgO_full_tmp_2734 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2735 CaMgO_full_tmp_2735 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2736 CaMgO_full_tmp_2736 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2737 CaMgO_full_tmp_2737 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2738 CaMgO_full_tmp_2738 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2739 CaMgO_full_tmp_2739 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2740 CaMgO_full_tmp_2740 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2741 CaMgO_full_tmp_2741 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2742 CaMgO_full_tmp_2742 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2743 CaMgO_full_tmp_2743 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2744 CaMgO_full_tmp_2744 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2745 CaMgO_full_tmp_2745 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2746 CaMgO_full_tmp_2746 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2747 CaMgO_full_tmp_2747 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2748 CaMgO_full_tmp_2748 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2749 CaMgO_full_tmp_2749 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2750 CaMgO_full_tmp_2750 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2751 CaMgO_full_tmp_2751 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2752 CaMgO_full_tmp_2752 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2753 CaMgO_full_tmp_2753 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2754 CaMgO_full_tmp_2754 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2755 CaMgO_full_tmp_2755 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2756 CaMgO_full_tmp_2756 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2757 CaMgO_full_tmp_2757 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2758 CaMgO_full_tmp_2758 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2759 CaMgO_full_tmp_2759 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2760 CaMgO_full_tmp_2760 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2761 CaMgO_full_tmp_2761 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2762 CaMgO_full_tmp_2762 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2763 CaMgO_full_tmp_2763 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2764 CaMgO_full_tmp_2764 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2765 CaMgO_full_tmp_2765 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2766 CaMgO_full_tmp_2766 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2767 CaMgO_full_tmp_2767 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2768 CaMgO_full_tmp_2768 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2769 CaMgO_full_tmp_2769 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2770 CaMgO_full_tmp_2770 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2771 CaMgO_full_tmp_2771 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2772 CaMgO_full_tmp_2772 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2773 CaMgO_full_tmp_2773 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2774 CaMgO_full_tmp_2774 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2775 CaMgO_full_tmp_2775 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2776 CaMgO_full_tmp_2776 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2777 CaMgO_full_tmp_2777 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2778 CaMgO_full_tmp_2778 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2779 CaMgO_full_tmp_2779 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2780 CaMgO_full_tmp_2780 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2781 CaMgO_full_tmp_2781 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2782 CaMgO_full_tmp_2782 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2783 CaMgO_full_tmp_2783 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2784 CaMgO_full_tmp_2784 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2785 CaMgO_full_tmp_2785 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2786 CaMgO_full_tmp_2786 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2787 CaMgO_full_tmp_2787 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2788 CaMgO_full_tmp_2788 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2789 CaMgO_full_tmp_2789 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2790 CaMgO_full_tmp_2790 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2791 CaMgO_full_tmp_2791 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2792 CaMgO_full_tmp_2792 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2793 CaMgO_full_tmp_2793 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2794 CaMgO_full_tmp_2794 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2795 CaMgO_full_tmp_2795 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2796 CaMgO_full_tmp_2796 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2797 CaMgO_full_tmp_2797 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2798 CaMgO_full_tmp_2798 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2799 CaMgO_full_tmp_2799 &

wait
mv CaMgO_full_tmp_2700.f9 CaMgO_full_2700.f9
mv CaMgO_full_tmp_2700.gui CaMgO_full_2700.gui
mv CaMgO_full_tmp_2701.f9 CaMgO_full_2701.f9
mv CaMgO_full_tmp_2701.gui CaMgO_full_2701.gui
mv CaMgO_full_tmp_2702.f9 CaMgO_full_2702.f9
mv CaMgO_full_tmp_2702.gui CaMgO_full_2702.gui
mv CaMgO_full_tmp_2703.f9 CaMgO_full_2703.f9
mv CaMgO_full_tmp_2703.gui CaMgO_full_2703.gui
mv CaMgO_full_tmp_2704.f9 CaMgO_full_2704.f9
mv CaMgO_full_tmp_2704.gui CaMgO_full_2704.gui
mv CaMgO_full_tmp_2705.f9 CaMgO_full_2705.f9
mv CaMgO_full_tmp_2705.gui CaMgO_full_2705.gui
mv CaMgO_full_tmp_2706.f9 CaMgO_full_2706.f9
mv CaMgO_full_tmp_2706.gui CaMgO_full_2706.gui
mv CaMgO_full_tmp_2707.f9 CaMgO_full_2707.f9
mv CaMgO_full_tmp_2707.gui CaMgO_full_2707.gui
mv CaMgO_full_tmp_2708.f9 CaMgO_full_2708.f9
mv CaMgO_full_tmp_2708.gui CaMgO_full_2708.gui
mv CaMgO_full_tmp_2709.f9 CaMgO_full_2709.f9
mv CaMgO_full_tmp_2709.gui CaMgO_full_2709.gui
mv CaMgO_full_tmp_2710.f9 CaMgO_full_2710.f9
mv CaMgO_full_tmp_2710.gui CaMgO_full_2710.gui
mv CaMgO_full_tmp_2711.f9 CaMgO_full_2711.f9
mv CaMgO_full_tmp_2711.gui CaMgO_full_2711.gui
mv CaMgO_full_tmp_2712.f9 CaMgO_full_2712.f9
mv CaMgO_full_tmp_2712.gui CaMgO_full_2712.gui
mv CaMgO_full_tmp_2713.f9 CaMgO_full_2713.f9
mv CaMgO_full_tmp_2713.gui CaMgO_full_2713.gui
mv CaMgO_full_tmp_2714.f9 CaMgO_full_2714.f9
mv CaMgO_full_tmp_2714.gui CaMgO_full_2714.gui
mv CaMgO_full_tmp_2715.f9 CaMgO_full_2715.f9
mv CaMgO_full_tmp_2715.gui CaMgO_full_2715.gui
mv CaMgO_full_tmp_2716.f9 CaMgO_full_2716.f9
mv CaMgO_full_tmp_2716.gui CaMgO_full_2716.gui
mv CaMgO_full_tmp_2717.f9 CaMgO_full_2717.f9
mv CaMgO_full_tmp_2717.gui CaMgO_full_2717.gui
mv CaMgO_full_tmp_2718.f9 CaMgO_full_2718.f9
mv CaMgO_full_tmp_2718.gui CaMgO_full_2718.gui
mv CaMgO_full_tmp_2719.f9 CaMgO_full_2719.f9
mv CaMgO_full_tmp_2719.gui CaMgO_full_2719.gui
mv CaMgO_full_tmp_2720.f9 CaMgO_full_2720.f9
mv CaMgO_full_tmp_2720.gui CaMgO_full_2720.gui
mv CaMgO_full_tmp_2721.f9 CaMgO_full_2721.f9
mv CaMgO_full_tmp_2721.gui CaMgO_full_2721.gui
mv CaMgO_full_tmp_2722.f9 CaMgO_full_2722.f9
mv CaMgO_full_tmp_2722.gui CaMgO_full_2722.gui
mv CaMgO_full_tmp_2723.f9 CaMgO_full_2723.f9
mv CaMgO_full_tmp_2723.gui CaMgO_full_2723.gui
mv CaMgO_full_tmp_2724.f9 CaMgO_full_2724.f9
mv CaMgO_full_tmp_2724.gui CaMgO_full_2724.gui
mv CaMgO_full_tmp_2725.f9 CaMgO_full_2725.f9
mv CaMgO_full_tmp_2725.gui CaMgO_full_2725.gui
mv CaMgO_full_tmp_2726.f9 CaMgO_full_2726.f9
mv CaMgO_full_tmp_2726.gui CaMgO_full_2726.gui
mv CaMgO_full_tmp_2727.f9 CaMgO_full_2727.f9
mv CaMgO_full_tmp_2727.gui CaMgO_full_2727.gui
mv CaMgO_full_tmp_2728.f9 CaMgO_full_2728.f9
mv CaMgO_full_tmp_2728.gui CaMgO_full_2728.gui
mv CaMgO_full_tmp_2729.f9 CaMgO_full_2729.f9
mv CaMgO_full_tmp_2729.gui CaMgO_full_2729.gui
mv CaMgO_full_tmp_2730.f9 CaMgO_full_2730.f9
mv CaMgO_full_tmp_2730.gui CaMgO_full_2730.gui
mv CaMgO_full_tmp_2731.f9 CaMgO_full_2731.f9
mv CaMgO_full_tmp_2731.gui CaMgO_full_2731.gui
mv CaMgO_full_tmp_2732.f9 CaMgO_full_2732.f9
mv CaMgO_full_tmp_2732.gui CaMgO_full_2732.gui
mv CaMgO_full_tmp_2733.f9 CaMgO_full_2733.f9
mv CaMgO_full_tmp_2733.gui CaMgO_full_2733.gui
mv CaMgO_full_tmp_2734.f9 CaMgO_full_2734.f9
mv CaMgO_full_tmp_2734.gui CaMgO_full_2734.gui
mv CaMgO_full_tmp_2735.f9 CaMgO_full_2735.f9
mv CaMgO_full_tmp_2735.gui CaMgO_full_2735.gui
mv CaMgO_full_tmp_2736.f9 CaMgO_full_2736.f9
mv CaMgO_full_tmp_2736.gui CaMgO_full_2736.gui
mv CaMgO_full_tmp_2737.f9 CaMgO_full_2737.f9
mv CaMgO_full_tmp_2737.gui CaMgO_full_2737.gui
mv CaMgO_full_tmp_2738.f9 CaMgO_full_2738.f9
mv CaMgO_full_tmp_2738.gui CaMgO_full_2738.gui
mv CaMgO_full_tmp_2739.f9 CaMgO_full_2739.f9
mv CaMgO_full_tmp_2739.gui CaMgO_full_2739.gui
mv CaMgO_full_tmp_2740.f9 CaMgO_full_2740.f9
mv CaMgO_full_tmp_2740.gui CaMgO_full_2740.gui
mv CaMgO_full_tmp_2741.f9 CaMgO_full_2741.f9
mv CaMgO_full_tmp_2741.gui CaMgO_full_2741.gui
mv CaMgO_full_tmp_2742.f9 CaMgO_full_2742.f9
mv CaMgO_full_tmp_2742.gui CaMgO_full_2742.gui
mv CaMgO_full_tmp_2743.f9 CaMgO_full_2743.f9
mv CaMgO_full_tmp_2743.gui CaMgO_full_2743.gui
mv CaMgO_full_tmp_2744.f9 CaMgO_full_2744.f9
mv CaMgO_full_tmp_2744.gui CaMgO_full_2744.gui
mv CaMgO_full_tmp_2745.f9 CaMgO_full_2745.f9
mv CaMgO_full_tmp_2745.gui CaMgO_full_2745.gui
mv CaMgO_full_tmp_2746.f9 CaMgO_full_2746.f9
mv CaMgO_full_tmp_2746.gui CaMgO_full_2746.gui
mv CaMgO_full_tmp_2747.f9 CaMgO_full_2747.f9
mv CaMgO_full_tmp_2747.gui CaMgO_full_2747.gui
mv CaMgO_full_tmp_2748.f9 CaMgO_full_2748.f9
mv CaMgO_full_tmp_2748.gui CaMgO_full_2748.gui
mv CaMgO_full_tmp_2749.f9 CaMgO_full_2749.f9
mv CaMgO_full_tmp_2749.gui CaMgO_full_2749.gui
mv CaMgO_full_tmp_2750.f9 CaMgO_full_2750.f9
mv CaMgO_full_tmp_2750.gui CaMgO_full_2750.gui
mv CaMgO_full_tmp_2751.f9 CaMgO_full_2751.f9
mv CaMgO_full_tmp_2751.gui CaMgO_full_2751.gui
mv CaMgO_full_tmp_2752.f9 CaMgO_full_2752.f9
mv CaMgO_full_tmp_2752.gui CaMgO_full_2752.gui
mv CaMgO_full_tmp_2753.f9 CaMgO_full_2753.f9
mv CaMgO_full_tmp_2753.gui CaMgO_full_2753.gui
mv CaMgO_full_tmp_2754.f9 CaMgO_full_2754.f9
mv CaMgO_full_tmp_2754.gui CaMgO_full_2754.gui
mv CaMgO_full_tmp_2755.f9 CaMgO_full_2755.f9
mv CaMgO_full_tmp_2755.gui CaMgO_full_2755.gui
mv CaMgO_full_tmp_2756.f9 CaMgO_full_2756.f9
mv CaMgO_full_tmp_2756.gui CaMgO_full_2756.gui
mv CaMgO_full_tmp_2757.f9 CaMgO_full_2757.f9
mv CaMgO_full_tmp_2757.gui CaMgO_full_2757.gui
mv CaMgO_full_tmp_2758.f9 CaMgO_full_2758.f9
mv CaMgO_full_tmp_2758.gui CaMgO_full_2758.gui
mv CaMgO_full_tmp_2759.f9 CaMgO_full_2759.f9
mv CaMgO_full_tmp_2759.gui CaMgO_full_2759.gui
mv CaMgO_full_tmp_2760.f9 CaMgO_full_2760.f9
mv CaMgO_full_tmp_2760.gui CaMgO_full_2760.gui
mv CaMgO_full_tmp_2761.f9 CaMgO_full_2761.f9
mv CaMgO_full_tmp_2761.gui CaMgO_full_2761.gui
mv CaMgO_full_tmp_2762.f9 CaMgO_full_2762.f9
mv CaMgO_full_tmp_2762.gui CaMgO_full_2762.gui
mv CaMgO_full_tmp_2763.f9 CaMgO_full_2763.f9
mv CaMgO_full_tmp_2763.gui CaMgO_full_2763.gui
mv CaMgO_full_tmp_2764.f9 CaMgO_full_2764.f9
mv CaMgO_full_tmp_2764.gui CaMgO_full_2764.gui
mv CaMgO_full_tmp_2765.f9 CaMgO_full_2765.f9
mv CaMgO_full_tmp_2765.gui CaMgO_full_2765.gui
mv CaMgO_full_tmp_2766.f9 CaMgO_full_2766.f9
mv CaMgO_full_tmp_2766.gui CaMgO_full_2766.gui
mv CaMgO_full_tmp_2767.f9 CaMgO_full_2767.f9
mv CaMgO_full_tmp_2767.gui CaMgO_full_2767.gui
mv CaMgO_full_tmp_2768.f9 CaMgO_full_2768.f9
mv CaMgO_full_tmp_2768.gui CaMgO_full_2768.gui
mv CaMgO_full_tmp_2769.f9 CaMgO_full_2769.f9
mv CaMgO_full_tmp_2769.gui CaMgO_full_2769.gui
mv CaMgO_full_tmp_2770.f9 CaMgO_full_2770.f9
mv CaMgO_full_tmp_2770.gui CaMgO_full_2770.gui
mv CaMgO_full_tmp_2771.f9 CaMgO_full_2771.f9
mv CaMgO_full_tmp_2771.gui CaMgO_full_2771.gui
mv CaMgO_full_tmp_2772.f9 CaMgO_full_2772.f9
mv CaMgO_full_tmp_2772.gui CaMgO_full_2772.gui
mv CaMgO_full_tmp_2773.f9 CaMgO_full_2773.f9
mv CaMgO_full_tmp_2773.gui CaMgO_full_2773.gui
mv CaMgO_full_tmp_2774.f9 CaMgO_full_2774.f9
mv CaMgO_full_tmp_2774.gui CaMgO_full_2774.gui
mv CaMgO_full_tmp_2775.f9 CaMgO_full_2775.f9
mv CaMgO_full_tmp_2775.gui CaMgO_full_2775.gui
mv CaMgO_full_tmp_2776.f9 CaMgO_full_2776.f9
mv CaMgO_full_tmp_2776.gui CaMgO_full_2776.gui
mv CaMgO_full_tmp_2777.f9 CaMgO_full_2777.f9
mv CaMgO_full_tmp_2777.gui CaMgO_full_2777.gui
mv CaMgO_full_tmp_2778.f9 CaMgO_full_2778.f9
mv CaMgO_full_tmp_2778.gui CaMgO_full_2778.gui
mv CaMgO_full_tmp_2779.f9 CaMgO_full_2779.f9
mv CaMgO_full_tmp_2779.gui CaMgO_full_2779.gui
mv CaMgO_full_tmp_2780.f9 CaMgO_full_2780.f9
mv CaMgO_full_tmp_2780.gui CaMgO_full_2780.gui
mv CaMgO_full_tmp_2781.f9 CaMgO_full_2781.f9
mv CaMgO_full_tmp_2781.gui CaMgO_full_2781.gui
mv CaMgO_full_tmp_2782.f9 CaMgO_full_2782.f9
mv CaMgO_full_tmp_2782.gui CaMgO_full_2782.gui
mv CaMgO_full_tmp_2783.f9 CaMgO_full_2783.f9
mv CaMgO_full_tmp_2783.gui CaMgO_full_2783.gui
mv CaMgO_full_tmp_2784.f9 CaMgO_full_2784.f9
mv CaMgO_full_tmp_2784.gui CaMgO_full_2784.gui
mv CaMgO_full_tmp_2785.f9 CaMgO_full_2785.f9
mv CaMgO_full_tmp_2785.gui CaMgO_full_2785.gui
mv CaMgO_full_tmp_2786.f9 CaMgO_full_2786.f9
mv CaMgO_full_tmp_2786.gui CaMgO_full_2786.gui
mv CaMgO_full_tmp_2787.f9 CaMgO_full_2787.f9
mv CaMgO_full_tmp_2787.gui CaMgO_full_2787.gui
mv CaMgO_full_tmp_2788.f9 CaMgO_full_2788.f9
mv CaMgO_full_tmp_2788.gui CaMgO_full_2788.gui
mv CaMgO_full_tmp_2789.f9 CaMgO_full_2789.f9
mv CaMgO_full_tmp_2789.gui CaMgO_full_2789.gui
mv CaMgO_full_tmp_2790.f9 CaMgO_full_2790.f9
mv CaMgO_full_tmp_2790.gui CaMgO_full_2790.gui
mv CaMgO_full_tmp_2791.f9 CaMgO_full_2791.f9
mv CaMgO_full_tmp_2791.gui CaMgO_full_2791.gui
mv CaMgO_full_tmp_2792.f9 CaMgO_full_2792.f9
mv CaMgO_full_tmp_2792.gui CaMgO_full_2792.gui
mv CaMgO_full_tmp_2793.f9 CaMgO_full_2793.f9
mv CaMgO_full_tmp_2793.gui CaMgO_full_2793.gui
mv CaMgO_full_tmp_2794.f9 CaMgO_full_2794.f9
mv CaMgO_full_tmp_2794.gui CaMgO_full_2794.gui
mv CaMgO_full_tmp_2795.f9 CaMgO_full_2795.f9
mv CaMgO_full_tmp_2795.gui CaMgO_full_2795.gui
mv CaMgO_full_tmp_2796.f9 CaMgO_full_2796.f9
mv CaMgO_full_tmp_2796.gui CaMgO_full_2796.gui
mv CaMgO_full_tmp_2797.f9 CaMgO_full_2797.f9
mv CaMgO_full_tmp_2797.gui CaMgO_full_2797.gui
mv CaMgO_full_tmp_2798.f9 CaMgO_full_2798.f9
mv CaMgO_full_tmp_2798.gui CaMgO_full_2798.gui
mv CaMgO_full_tmp_2799.f9 CaMgO_full_2799.f9
mv CaMgO_full_tmp_2799.gui CaMgO_full_2799.gui
