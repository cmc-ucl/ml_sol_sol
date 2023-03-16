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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2899


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2900


