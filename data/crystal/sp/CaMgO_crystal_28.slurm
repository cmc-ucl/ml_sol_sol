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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2701
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2701


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2702
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2702


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2703
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2703


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2704
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2704


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2705
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2705


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2706
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2706


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2707
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2707


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2708
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2708


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2709
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2709


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2710
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2710


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2711
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2711


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2712
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2712


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2713
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2713


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2714
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2714


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2715
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2715


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2716
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2716


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2717
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2717


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2718
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2718


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2719
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2719


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2720
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2720


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2721
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2721


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2722
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2722


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2723
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2723


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2724
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2724


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2725
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2725


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2726
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2726


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2727
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2727


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2728
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2728


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2729
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2729


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2730
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2730


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2731
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2731


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2732
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2732


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2733
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2733


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2734
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2734


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2735
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2735


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2736
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2736


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2737
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2737


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2738
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2738


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2739
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2739


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2740
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2740


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2741
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2741


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2742
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2742


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2743
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2743


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2744
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2744


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2745
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2745


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2746
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2746


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2747
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2747


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2748
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2748


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2749
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2749


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2750
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2750


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2751
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2751


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2752
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2752


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2753
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2753


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2754
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2754


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2755
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2755


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2756
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2756


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2757
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2757


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2758
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2758


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2759
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2759


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2760
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2760


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2761
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2761


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2762
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2762


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2763
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2763


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2764
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2764


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2765
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2765


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2766
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2766


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2767
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2767


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2768
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2768


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2769
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2769


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2770
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2770


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2771
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2771


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2772
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2772


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2773
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2773


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2774
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2774


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2775
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2775


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2776
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2776


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2777
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2777


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2778
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2778


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2779
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2779


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2780
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2780


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2781
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2781


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2782
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2782


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2783
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2783


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2784
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2784


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2785
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2785


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2786
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2786


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2787
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2787


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2788
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2788


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2789
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2789


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2790
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2790


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2791
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2791


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2792
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2792


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2793
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2793


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2794
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2794


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2795
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2795


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2796
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2796


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2797
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2797


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2798
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2798


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2799
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2799


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2800
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2800


