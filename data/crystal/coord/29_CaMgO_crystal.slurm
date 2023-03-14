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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2801 CaMgO_coord_2801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2802 CaMgO_coord_2802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2803 CaMgO_coord_2803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2804 CaMgO_coord_2804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2805 CaMgO_coord_2805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2806 CaMgO_coord_2806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2807 CaMgO_coord_2807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2808 CaMgO_coord_2808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2809 CaMgO_coord_2809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2810 CaMgO_coord_2810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2811 CaMgO_coord_2811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2812 CaMgO_coord_2812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2813 CaMgO_coord_2813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2814 CaMgO_coord_2814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2815 CaMgO_coord_2815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2816 CaMgO_coord_2816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2817 CaMgO_coord_2817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2818 CaMgO_coord_2818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2819 CaMgO_coord_2819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2820 CaMgO_coord_2820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2821 CaMgO_coord_2821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2822 CaMgO_coord_2822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2823 CaMgO_coord_2823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2824 CaMgO_coord_2824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2825 CaMgO_coord_2825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2826 CaMgO_coord_2826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2827 CaMgO_coord_2827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2828 CaMgO_coord_2828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2829 CaMgO_coord_2829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2830 CaMgO_coord_2830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2831 CaMgO_coord_2831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2832 CaMgO_coord_2832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2833 CaMgO_coord_2833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2834 CaMgO_coord_2834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2835 CaMgO_coord_2835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2836 CaMgO_coord_2836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2837 CaMgO_coord_2837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2838 CaMgO_coord_2838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2839 CaMgO_coord_2839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2840 CaMgO_coord_2840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2841 CaMgO_coord_2841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2842 CaMgO_coord_2842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2843 CaMgO_coord_2843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2844 CaMgO_coord_2844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2845 CaMgO_coord_2845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2846 CaMgO_coord_2846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2847 CaMgO_coord_2847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2848 CaMgO_coord_2848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2849 CaMgO_coord_2849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2850 CaMgO_coord_2850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2851 CaMgO_coord_2851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2852 CaMgO_coord_2852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2853 CaMgO_coord_2853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2854 CaMgO_coord_2854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2855 CaMgO_coord_2855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2856 CaMgO_coord_2856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2857 CaMgO_coord_2857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2858 CaMgO_coord_2858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2859 CaMgO_coord_2859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2860 CaMgO_coord_2860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2861 CaMgO_coord_2861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2862 CaMgO_coord_2862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2863 CaMgO_coord_2863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2864 CaMgO_coord_2864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2865 CaMgO_coord_2865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2866 CaMgO_coord_2866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2867 CaMgO_coord_2867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2868 CaMgO_coord_2868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2869 CaMgO_coord_2869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2870 CaMgO_coord_2870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2871 CaMgO_coord_2871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2872 CaMgO_coord_2872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2873 CaMgO_coord_2873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2874 CaMgO_coord_2874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2875 CaMgO_coord_2875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2876 CaMgO_coord_2876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2877 CaMgO_coord_2877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2878 CaMgO_coord_2878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2879 CaMgO_coord_2879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2880 CaMgO_coord_2880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2881 CaMgO_coord_2881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2882 CaMgO_coord_2882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2883 CaMgO_coord_2883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2884 CaMgO_coord_2884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2885 CaMgO_coord_2885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2886 CaMgO_coord_2886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2887 CaMgO_coord_2887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2888 CaMgO_coord_2888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2889 CaMgO_coord_2889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2890 CaMgO_coord_2890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2891 CaMgO_coord_2891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2892 CaMgO_coord_2892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2893 CaMgO_coord_2893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2894 CaMgO_coord_2894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2895 CaMgO_coord_2895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2896 CaMgO_coord_2896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2897 CaMgO_coord_2897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2898 CaMgO_coord_2898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2899 CaMgO_coord_2899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2899


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2900 CaMgO_coord_2900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2900


