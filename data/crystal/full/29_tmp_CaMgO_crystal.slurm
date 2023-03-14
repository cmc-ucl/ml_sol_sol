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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2800 CaMgO_full_tmp_2800 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2801 CaMgO_full_tmp_2801 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2802 CaMgO_full_tmp_2802 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2803 CaMgO_full_tmp_2803 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2804 CaMgO_full_tmp_2804 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2805 CaMgO_full_tmp_2805 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2806 CaMgO_full_tmp_2806 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2807 CaMgO_full_tmp_2807 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2808 CaMgO_full_tmp_2808 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2809 CaMgO_full_tmp_2809 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2810 CaMgO_full_tmp_2810 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2811 CaMgO_full_tmp_2811 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2812 CaMgO_full_tmp_2812 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2813 CaMgO_full_tmp_2813 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2814 CaMgO_full_tmp_2814 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2815 CaMgO_full_tmp_2815 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2816 CaMgO_full_tmp_2816 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2817 CaMgO_full_tmp_2817 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2818 CaMgO_full_tmp_2818 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2819 CaMgO_full_tmp_2819 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2820 CaMgO_full_tmp_2820 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2821 CaMgO_full_tmp_2821 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2822 CaMgO_full_tmp_2822 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2823 CaMgO_full_tmp_2823 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2824 CaMgO_full_tmp_2824 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2825 CaMgO_full_tmp_2825 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2826 CaMgO_full_tmp_2826 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2827 CaMgO_full_tmp_2827 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2828 CaMgO_full_tmp_2828 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2829 CaMgO_full_tmp_2829 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2830 CaMgO_full_tmp_2830 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2831 CaMgO_full_tmp_2831 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2832 CaMgO_full_tmp_2832 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2833 CaMgO_full_tmp_2833 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2834 CaMgO_full_tmp_2834 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2835 CaMgO_full_tmp_2835 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2836 CaMgO_full_tmp_2836 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2837 CaMgO_full_tmp_2837 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2838 CaMgO_full_tmp_2838 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2839 CaMgO_full_tmp_2839 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2840 CaMgO_full_tmp_2840 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2841 CaMgO_full_tmp_2841 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2842 CaMgO_full_tmp_2842 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2843 CaMgO_full_tmp_2843 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2844 CaMgO_full_tmp_2844 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2845 CaMgO_full_tmp_2845 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2846 CaMgO_full_tmp_2846 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2847 CaMgO_full_tmp_2847 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2848 CaMgO_full_tmp_2848 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2849 CaMgO_full_tmp_2849 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2850 CaMgO_full_tmp_2850 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2851 CaMgO_full_tmp_2851 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2852 CaMgO_full_tmp_2852 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2853 CaMgO_full_tmp_2853 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2854 CaMgO_full_tmp_2854 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2855 CaMgO_full_tmp_2855 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2856 CaMgO_full_tmp_2856 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2857 CaMgO_full_tmp_2857 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2858 CaMgO_full_tmp_2858 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2859 CaMgO_full_tmp_2859 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2860 CaMgO_full_tmp_2860 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2861 CaMgO_full_tmp_2861 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2862 CaMgO_full_tmp_2862 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2863 CaMgO_full_tmp_2863 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2864 CaMgO_full_tmp_2864 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2865 CaMgO_full_tmp_2865 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2866 CaMgO_full_tmp_2866 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2867 CaMgO_full_tmp_2867 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2868 CaMgO_full_tmp_2868 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2869 CaMgO_full_tmp_2869 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2870 CaMgO_full_tmp_2870 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2871 CaMgO_full_tmp_2871 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2872 CaMgO_full_tmp_2872 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2873 CaMgO_full_tmp_2873 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2874 CaMgO_full_tmp_2874 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2875 CaMgO_full_tmp_2875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2876 CaMgO_full_tmp_2876 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2877 CaMgO_full_tmp_2877 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2878 CaMgO_full_tmp_2878 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2879 CaMgO_full_tmp_2879 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2880 CaMgO_full_tmp_2880 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2881 CaMgO_full_tmp_2881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2882 CaMgO_full_tmp_2882 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2883 CaMgO_full_tmp_2883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2884 CaMgO_full_tmp_2884 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2885 CaMgO_full_tmp_2885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2886 CaMgO_full_tmp_2886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2887 CaMgO_full_tmp_2887 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2888 CaMgO_full_tmp_2888 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2889 CaMgO_full_tmp_2889 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2890 CaMgO_full_tmp_2890 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2891 CaMgO_full_tmp_2891 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2892 CaMgO_full_tmp_2892 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2893 CaMgO_full_tmp_2893 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2894 CaMgO_full_tmp_2894 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2895 CaMgO_full_tmp_2895 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2896 CaMgO_full_tmp_2896 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2897 CaMgO_full_tmp_2897 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2898 CaMgO_full_tmp_2898 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2899 CaMgO_full_tmp_2899 &

wait
mv CaMgO_full_tmp_2800.f9 CaMgO_full_2800.f9
mv CaMgO_full_tmp_2800.gui CaMgO_full_2800.gui
mv CaMgO_full_tmp_2801.f9 CaMgO_full_2801.f9
mv CaMgO_full_tmp_2801.gui CaMgO_full_2801.gui
mv CaMgO_full_tmp_2802.f9 CaMgO_full_2802.f9
mv CaMgO_full_tmp_2802.gui CaMgO_full_2802.gui
mv CaMgO_full_tmp_2803.f9 CaMgO_full_2803.f9
mv CaMgO_full_tmp_2803.gui CaMgO_full_2803.gui
mv CaMgO_full_tmp_2804.f9 CaMgO_full_2804.f9
mv CaMgO_full_tmp_2804.gui CaMgO_full_2804.gui
mv CaMgO_full_tmp_2805.f9 CaMgO_full_2805.f9
mv CaMgO_full_tmp_2805.gui CaMgO_full_2805.gui
mv CaMgO_full_tmp_2806.f9 CaMgO_full_2806.f9
mv CaMgO_full_tmp_2806.gui CaMgO_full_2806.gui
mv CaMgO_full_tmp_2807.f9 CaMgO_full_2807.f9
mv CaMgO_full_tmp_2807.gui CaMgO_full_2807.gui
mv CaMgO_full_tmp_2808.f9 CaMgO_full_2808.f9
mv CaMgO_full_tmp_2808.gui CaMgO_full_2808.gui
mv CaMgO_full_tmp_2809.f9 CaMgO_full_2809.f9
mv CaMgO_full_tmp_2809.gui CaMgO_full_2809.gui
mv CaMgO_full_tmp_2810.f9 CaMgO_full_2810.f9
mv CaMgO_full_tmp_2810.gui CaMgO_full_2810.gui
mv CaMgO_full_tmp_2811.f9 CaMgO_full_2811.f9
mv CaMgO_full_tmp_2811.gui CaMgO_full_2811.gui
mv CaMgO_full_tmp_2812.f9 CaMgO_full_2812.f9
mv CaMgO_full_tmp_2812.gui CaMgO_full_2812.gui
mv CaMgO_full_tmp_2813.f9 CaMgO_full_2813.f9
mv CaMgO_full_tmp_2813.gui CaMgO_full_2813.gui
mv CaMgO_full_tmp_2814.f9 CaMgO_full_2814.f9
mv CaMgO_full_tmp_2814.gui CaMgO_full_2814.gui
mv CaMgO_full_tmp_2815.f9 CaMgO_full_2815.f9
mv CaMgO_full_tmp_2815.gui CaMgO_full_2815.gui
mv CaMgO_full_tmp_2816.f9 CaMgO_full_2816.f9
mv CaMgO_full_tmp_2816.gui CaMgO_full_2816.gui
mv CaMgO_full_tmp_2817.f9 CaMgO_full_2817.f9
mv CaMgO_full_tmp_2817.gui CaMgO_full_2817.gui
mv CaMgO_full_tmp_2818.f9 CaMgO_full_2818.f9
mv CaMgO_full_tmp_2818.gui CaMgO_full_2818.gui
mv CaMgO_full_tmp_2819.f9 CaMgO_full_2819.f9
mv CaMgO_full_tmp_2819.gui CaMgO_full_2819.gui
mv CaMgO_full_tmp_2820.f9 CaMgO_full_2820.f9
mv CaMgO_full_tmp_2820.gui CaMgO_full_2820.gui
mv CaMgO_full_tmp_2821.f9 CaMgO_full_2821.f9
mv CaMgO_full_tmp_2821.gui CaMgO_full_2821.gui
mv CaMgO_full_tmp_2822.f9 CaMgO_full_2822.f9
mv CaMgO_full_tmp_2822.gui CaMgO_full_2822.gui
mv CaMgO_full_tmp_2823.f9 CaMgO_full_2823.f9
mv CaMgO_full_tmp_2823.gui CaMgO_full_2823.gui
mv CaMgO_full_tmp_2824.f9 CaMgO_full_2824.f9
mv CaMgO_full_tmp_2824.gui CaMgO_full_2824.gui
mv CaMgO_full_tmp_2825.f9 CaMgO_full_2825.f9
mv CaMgO_full_tmp_2825.gui CaMgO_full_2825.gui
mv CaMgO_full_tmp_2826.f9 CaMgO_full_2826.f9
mv CaMgO_full_tmp_2826.gui CaMgO_full_2826.gui
mv CaMgO_full_tmp_2827.f9 CaMgO_full_2827.f9
mv CaMgO_full_tmp_2827.gui CaMgO_full_2827.gui
mv CaMgO_full_tmp_2828.f9 CaMgO_full_2828.f9
mv CaMgO_full_tmp_2828.gui CaMgO_full_2828.gui
mv CaMgO_full_tmp_2829.f9 CaMgO_full_2829.f9
mv CaMgO_full_tmp_2829.gui CaMgO_full_2829.gui
mv CaMgO_full_tmp_2830.f9 CaMgO_full_2830.f9
mv CaMgO_full_tmp_2830.gui CaMgO_full_2830.gui
mv CaMgO_full_tmp_2831.f9 CaMgO_full_2831.f9
mv CaMgO_full_tmp_2831.gui CaMgO_full_2831.gui
mv CaMgO_full_tmp_2832.f9 CaMgO_full_2832.f9
mv CaMgO_full_tmp_2832.gui CaMgO_full_2832.gui
mv CaMgO_full_tmp_2833.f9 CaMgO_full_2833.f9
mv CaMgO_full_tmp_2833.gui CaMgO_full_2833.gui
mv CaMgO_full_tmp_2834.f9 CaMgO_full_2834.f9
mv CaMgO_full_tmp_2834.gui CaMgO_full_2834.gui
mv CaMgO_full_tmp_2835.f9 CaMgO_full_2835.f9
mv CaMgO_full_tmp_2835.gui CaMgO_full_2835.gui
mv CaMgO_full_tmp_2836.f9 CaMgO_full_2836.f9
mv CaMgO_full_tmp_2836.gui CaMgO_full_2836.gui
mv CaMgO_full_tmp_2837.f9 CaMgO_full_2837.f9
mv CaMgO_full_tmp_2837.gui CaMgO_full_2837.gui
mv CaMgO_full_tmp_2838.f9 CaMgO_full_2838.f9
mv CaMgO_full_tmp_2838.gui CaMgO_full_2838.gui
mv CaMgO_full_tmp_2839.f9 CaMgO_full_2839.f9
mv CaMgO_full_tmp_2839.gui CaMgO_full_2839.gui
mv CaMgO_full_tmp_2840.f9 CaMgO_full_2840.f9
mv CaMgO_full_tmp_2840.gui CaMgO_full_2840.gui
mv CaMgO_full_tmp_2841.f9 CaMgO_full_2841.f9
mv CaMgO_full_tmp_2841.gui CaMgO_full_2841.gui
mv CaMgO_full_tmp_2842.f9 CaMgO_full_2842.f9
mv CaMgO_full_tmp_2842.gui CaMgO_full_2842.gui
mv CaMgO_full_tmp_2843.f9 CaMgO_full_2843.f9
mv CaMgO_full_tmp_2843.gui CaMgO_full_2843.gui
mv CaMgO_full_tmp_2844.f9 CaMgO_full_2844.f9
mv CaMgO_full_tmp_2844.gui CaMgO_full_2844.gui
mv CaMgO_full_tmp_2845.f9 CaMgO_full_2845.f9
mv CaMgO_full_tmp_2845.gui CaMgO_full_2845.gui
mv CaMgO_full_tmp_2846.f9 CaMgO_full_2846.f9
mv CaMgO_full_tmp_2846.gui CaMgO_full_2846.gui
mv CaMgO_full_tmp_2847.f9 CaMgO_full_2847.f9
mv CaMgO_full_tmp_2847.gui CaMgO_full_2847.gui
mv CaMgO_full_tmp_2848.f9 CaMgO_full_2848.f9
mv CaMgO_full_tmp_2848.gui CaMgO_full_2848.gui
mv CaMgO_full_tmp_2849.f9 CaMgO_full_2849.f9
mv CaMgO_full_tmp_2849.gui CaMgO_full_2849.gui
mv CaMgO_full_tmp_2850.f9 CaMgO_full_2850.f9
mv CaMgO_full_tmp_2850.gui CaMgO_full_2850.gui
mv CaMgO_full_tmp_2851.f9 CaMgO_full_2851.f9
mv CaMgO_full_tmp_2851.gui CaMgO_full_2851.gui
mv CaMgO_full_tmp_2852.f9 CaMgO_full_2852.f9
mv CaMgO_full_tmp_2852.gui CaMgO_full_2852.gui
mv CaMgO_full_tmp_2853.f9 CaMgO_full_2853.f9
mv CaMgO_full_tmp_2853.gui CaMgO_full_2853.gui
mv CaMgO_full_tmp_2854.f9 CaMgO_full_2854.f9
mv CaMgO_full_tmp_2854.gui CaMgO_full_2854.gui
mv CaMgO_full_tmp_2855.f9 CaMgO_full_2855.f9
mv CaMgO_full_tmp_2855.gui CaMgO_full_2855.gui
mv CaMgO_full_tmp_2856.f9 CaMgO_full_2856.f9
mv CaMgO_full_tmp_2856.gui CaMgO_full_2856.gui
mv CaMgO_full_tmp_2857.f9 CaMgO_full_2857.f9
mv CaMgO_full_tmp_2857.gui CaMgO_full_2857.gui
mv CaMgO_full_tmp_2858.f9 CaMgO_full_2858.f9
mv CaMgO_full_tmp_2858.gui CaMgO_full_2858.gui
mv CaMgO_full_tmp_2859.f9 CaMgO_full_2859.f9
mv CaMgO_full_tmp_2859.gui CaMgO_full_2859.gui
mv CaMgO_full_tmp_2860.f9 CaMgO_full_2860.f9
mv CaMgO_full_tmp_2860.gui CaMgO_full_2860.gui
mv CaMgO_full_tmp_2861.f9 CaMgO_full_2861.f9
mv CaMgO_full_tmp_2861.gui CaMgO_full_2861.gui
mv CaMgO_full_tmp_2862.f9 CaMgO_full_2862.f9
mv CaMgO_full_tmp_2862.gui CaMgO_full_2862.gui
mv CaMgO_full_tmp_2863.f9 CaMgO_full_2863.f9
mv CaMgO_full_tmp_2863.gui CaMgO_full_2863.gui
mv CaMgO_full_tmp_2864.f9 CaMgO_full_2864.f9
mv CaMgO_full_tmp_2864.gui CaMgO_full_2864.gui
mv CaMgO_full_tmp_2865.f9 CaMgO_full_2865.f9
mv CaMgO_full_tmp_2865.gui CaMgO_full_2865.gui
mv CaMgO_full_tmp_2866.f9 CaMgO_full_2866.f9
mv CaMgO_full_tmp_2866.gui CaMgO_full_2866.gui
mv CaMgO_full_tmp_2867.f9 CaMgO_full_2867.f9
mv CaMgO_full_tmp_2867.gui CaMgO_full_2867.gui
mv CaMgO_full_tmp_2868.f9 CaMgO_full_2868.f9
mv CaMgO_full_tmp_2868.gui CaMgO_full_2868.gui
mv CaMgO_full_tmp_2869.f9 CaMgO_full_2869.f9
mv CaMgO_full_tmp_2869.gui CaMgO_full_2869.gui
mv CaMgO_full_tmp_2870.f9 CaMgO_full_2870.f9
mv CaMgO_full_tmp_2870.gui CaMgO_full_2870.gui
mv CaMgO_full_tmp_2871.f9 CaMgO_full_2871.f9
mv CaMgO_full_tmp_2871.gui CaMgO_full_2871.gui
mv CaMgO_full_tmp_2872.f9 CaMgO_full_2872.f9
mv CaMgO_full_tmp_2872.gui CaMgO_full_2872.gui
mv CaMgO_full_tmp_2873.f9 CaMgO_full_2873.f9
mv CaMgO_full_tmp_2873.gui CaMgO_full_2873.gui
mv CaMgO_full_tmp_2874.f9 CaMgO_full_2874.f9
mv CaMgO_full_tmp_2874.gui CaMgO_full_2874.gui
mv CaMgO_full_tmp_2875.f9 CaMgO_full_2875.f9
mv CaMgO_full_tmp_2875.gui CaMgO_full_2875.gui
mv CaMgO_full_tmp_2876.f9 CaMgO_full_2876.f9
mv CaMgO_full_tmp_2876.gui CaMgO_full_2876.gui
mv CaMgO_full_tmp_2877.f9 CaMgO_full_2877.f9
mv CaMgO_full_tmp_2877.gui CaMgO_full_2877.gui
mv CaMgO_full_tmp_2878.f9 CaMgO_full_2878.f9
mv CaMgO_full_tmp_2878.gui CaMgO_full_2878.gui
mv CaMgO_full_tmp_2879.f9 CaMgO_full_2879.f9
mv CaMgO_full_tmp_2879.gui CaMgO_full_2879.gui
mv CaMgO_full_tmp_2880.f9 CaMgO_full_2880.f9
mv CaMgO_full_tmp_2880.gui CaMgO_full_2880.gui
mv CaMgO_full_tmp_2881.f9 CaMgO_full_2881.f9
mv CaMgO_full_tmp_2881.gui CaMgO_full_2881.gui
mv CaMgO_full_tmp_2882.f9 CaMgO_full_2882.f9
mv CaMgO_full_tmp_2882.gui CaMgO_full_2882.gui
mv CaMgO_full_tmp_2883.f9 CaMgO_full_2883.f9
mv CaMgO_full_tmp_2883.gui CaMgO_full_2883.gui
mv CaMgO_full_tmp_2884.f9 CaMgO_full_2884.f9
mv CaMgO_full_tmp_2884.gui CaMgO_full_2884.gui
mv CaMgO_full_tmp_2885.f9 CaMgO_full_2885.f9
mv CaMgO_full_tmp_2885.gui CaMgO_full_2885.gui
mv CaMgO_full_tmp_2886.f9 CaMgO_full_2886.f9
mv CaMgO_full_tmp_2886.gui CaMgO_full_2886.gui
mv CaMgO_full_tmp_2887.f9 CaMgO_full_2887.f9
mv CaMgO_full_tmp_2887.gui CaMgO_full_2887.gui
mv CaMgO_full_tmp_2888.f9 CaMgO_full_2888.f9
mv CaMgO_full_tmp_2888.gui CaMgO_full_2888.gui
mv CaMgO_full_tmp_2889.f9 CaMgO_full_2889.f9
mv CaMgO_full_tmp_2889.gui CaMgO_full_2889.gui
mv CaMgO_full_tmp_2890.f9 CaMgO_full_2890.f9
mv CaMgO_full_tmp_2890.gui CaMgO_full_2890.gui
mv CaMgO_full_tmp_2891.f9 CaMgO_full_2891.f9
mv CaMgO_full_tmp_2891.gui CaMgO_full_2891.gui
mv CaMgO_full_tmp_2892.f9 CaMgO_full_2892.f9
mv CaMgO_full_tmp_2892.gui CaMgO_full_2892.gui
mv CaMgO_full_tmp_2893.f9 CaMgO_full_2893.f9
mv CaMgO_full_tmp_2893.gui CaMgO_full_2893.gui
mv CaMgO_full_tmp_2894.f9 CaMgO_full_2894.f9
mv CaMgO_full_tmp_2894.gui CaMgO_full_2894.gui
mv CaMgO_full_tmp_2895.f9 CaMgO_full_2895.f9
mv CaMgO_full_tmp_2895.gui CaMgO_full_2895.gui
mv CaMgO_full_tmp_2896.f9 CaMgO_full_2896.f9
mv CaMgO_full_tmp_2896.gui CaMgO_full_2896.gui
mv CaMgO_full_tmp_2897.f9 CaMgO_full_2897.f9
mv CaMgO_full_tmp_2897.gui CaMgO_full_2897.gui
mv CaMgO_full_tmp_2898.f9 CaMgO_full_2898.f9
mv CaMgO_full_tmp_2898.gui CaMgO_full_2898.gui
mv CaMgO_full_tmp_2899.f9 CaMgO_full_2899.f9
mv CaMgO_full_tmp_2899.gui CaMgO_full_2899.gui
