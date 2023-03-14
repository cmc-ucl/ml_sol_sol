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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2900 CaMgO_full_tmp_2900 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2901 CaMgO_full_tmp_2901 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2902 CaMgO_full_tmp_2902 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2903 CaMgO_full_tmp_2903 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2904 CaMgO_full_tmp_2904 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2905 CaMgO_full_tmp_2905 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2906 CaMgO_full_tmp_2906 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2907 CaMgO_full_tmp_2907 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2908 CaMgO_full_tmp_2908 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2909 CaMgO_full_tmp_2909 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2910 CaMgO_full_tmp_2910 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2911 CaMgO_full_tmp_2911 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2912 CaMgO_full_tmp_2912 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2913 CaMgO_full_tmp_2913 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2914 CaMgO_full_tmp_2914 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2915 CaMgO_full_tmp_2915 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2916 CaMgO_full_tmp_2916 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2917 CaMgO_full_tmp_2917 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2918 CaMgO_full_tmp_2918 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2919 CaMgO_full_tmp_2919 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2920 CaMgO_full_tmp_2920 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2921 CaMgO_full_tmp_2921 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2922 CaMgO_full_tmp_2922 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2923 CaMgO_full_tmp_2923 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2924 CaMgO_full_tmp_2924 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2925 CaMgO_full_tmp_2925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2926 CaMgO_full_tmp_2926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2927 CaMgO_full_tmp_2927 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2928 CaMgO_full_tmp_2928 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2929 CaMgO_full_tmp_2929 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2930 CaMgO_full_tmp_2930 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2931 CaMgO_full_tmp_2931 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2932 CaMgO_full_tmp_2932 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2933 CaMgO_full_tmp_2933 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2934 CaMgO_full_tmp_2934 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2935 CaMgO_full_tmp_2935 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2936 CaMgO_full_tmp_2936 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2937 CaMgO_full_tmp_2937 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2938 CaMgO_full_tmp_2938 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2939 CaMgO_full_tmp_2939 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2940 CaMgO_full_tmp_2940 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2941 CaMgO_full_tmp_2941 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2942 CaMgO_full_tmp_2942 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2943 CaMgO_full_tmp_2943 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2944 CaMgO_full_tmp_2944 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2945 CaMgO_full_tmp_2945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2946 CaMgO_full_tmp_2946 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2947 CaMgO_full_tmp_2947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2948 CaMgO_full_tmp_2948 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2949 CaMgO_full_tmp_2949 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2950 CaMgO_full_tmp_2950 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2951 CaMgO_full_tmp_2951 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2952 CaMgO_full_tmp_2952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2953 CaMgO_full_tmp_2953 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2954 CaMgO_full_tmp_2954 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2955 CaMgO_full_tmp_2955 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2956 CaMgO_full_tmp_2956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2957 CaMgO_full_tmp_2957 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2958 CaMgO_full_tmp_2958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2959 CaMgO_full_tmp_2959 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2960 CaMgO_full_tmp_2960 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2961 CaMgO_full_tmp_2961 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2962 CaMgO_full_tmp_2962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2963 CaMgO_full_tmp_2963 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2964 CaMgO_full_tmp_2964 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2965 CaMgO_full_tmp_2965 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2966 CaMgO_full_tmp_2966 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2967 CaMgO_full_tmp_2967 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2968 CaMgO_full_tmp_2968 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2969 CaMgO_full_tmp_2969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2970 CaMgO_full_tmp_2970 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2971 CaMgO_full_tmp_2971 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2972 CaMgO_full_tmp_2972 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2973 CaMgO_full_tmp_2973 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2974 CaMgO_full_tmp_2974 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2975 CaMgO_full_tmp_2975 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2976 CaMgO_full_tmp_2976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2977 CaMgO_full_tmp_2977 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2978 CaMgO_full_tmp_2978 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2979 CaMgO_full_tmp_2979 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2980 CaMgO_full_tmp_2980 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2981 CaMgO_full_tmp_2981 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2982 CaMgO_full_tmp_2982 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2983 CaMgO_full_tmp_2983 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2984 CaMgO_full_tmp_2984 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2985 CaMgO_full_tmp_2985 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2986 CaMgO_full_tmp_2986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2987 CaMgO_full_tmp_2987 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2988 CaMgO_full_tmp_2988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2989 CaMgO_full_tmp_2989 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2990 CaMgO_full_tmp_2990 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2991 CaMgO_full_tmp_2991 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2992 CaMgO_full_tmp_2992 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2993 CaMgO_full_tmp_2993 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2994 CaMgO_full_tmp_2994 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2995 CaMgO_full_tmp_2995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2996 CaMgO_full_tmp_2996 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2997 CaMgO_full_tmp_2997 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2998 CaMgO_full_tmp_2998 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2999 CaMgO_full_tmp_2999 &

wait
mv CaMgO_full_tmp_2900.f9 CaMgO_full_2900.f9
mv CaMgO_full_tmp_2900.gui CaMgO_full_2900.gui
mv CaMgO_full_tmp_2901.f9 CaMgO_full_2901.f9
mv CaMgO_full_tmp_2901.gui CaMgO_full_2901.gui
mv CaMgO_full_tmp_2902.f9 CaMgO_full_2902.f9
mv CaMgO_full_tmp_2902.gui CaMgO_full_2902.gui
mv CaMgO_full_tmp_2903.f9 CaMgO_full_2903.f9
mv CaMgO_full_tmp_2903.gui CaMgO_full_2903.gui
mv CaMgO_full_tmp_2904.f9 CaMgO_full_2904.f9
mv CaMgO_full_tmp_2904.gui CaMgO_full_2904.gui
mv CaMgO_full_tmp_2905.f9 CaMgO_full_2905.f9
mv CaMgO_full_tmp_2905.gui CaMgO_full_2905.gui
mv CaMgO_full_tmp_2906.f9 CaMgO_full_2906.f9
mv CaMgO_full_tmp_2906.gui CaMgO_full_2906.gui
mv CaMgO_full_tmp_2907.f9 CaMgO_full_2907.f9
mv CaMgO_full_tmp_2907.gui CaMgO_full_2907.gui
mv CaMgO_full_tmp_2908.f9 CaMgO_full_2908.f9
mv CaMgO_full_tmp_2908.gui CaMgO_full_2908.gui
mv CaMgO_full_tmp_2909.f9 CaMgO_full_2909.f9
mv CaMgO_full_tmp_2909.gui CaMgO_full_2909.gui
mv CaMgO_full_tmp_2910.f9 CaMgO_full_2910.f9
mv CaMgO_full_tmp_2910.gui CaMgO_full_2910.gui
mv CaMgO_full_tmp_2911.f9 CaMgO_full_2911.f9
mv CaMgO_full_tmp_2911.gui CaMgO_full_2911.gui
mv CaMgO_full_tmp_2912.f9 CaMgO_full_2912.f9
mv CaMgO_full_tmp_2912.gui CaMgO_full_2912.gui
mv CaMgO_full_tmp_2913.f9 CaMgO_full_2913.f9
mv CaMgO_full_tmp_2913.gui CaMgO_full_2913.gui
mv CaMgO_full_tmp_2914.f9 CaMgO_full_2914.f9
mv CaMgO_full_tmp_2914.gui CaMgO_full_2914.gui
mv CaMgO_full_tmp_2915.f9 CaMgO_full_2915.f9
mv CaMgO_full_tmp_2915.gui CaMgO_full_2915.gui
mv CaMgO_full_tmp_2916.f9 CaMgO_full_2916.f9
mv CaMgO_full_tmp_2916.gui CaMgO_full_2916.gui
mv CaMgO_full_tmp_2917.f9 CaMgO_full_2917.f9
mv CaMgO_full_tmp_2917.gui CaMgO_full_2917.gui
mv CaMgO_full_tmp_2918.f9 CaMgO_full_2918.f9
mv CaMgO_full_tmp_2918.gui CaMgO_full_2918.gui
mv CaMgO_full_tmp_2919.f9 CaMgO_full_2919.f9
mv CaMgO_full_tmp_2919.gui CaMgO_full_2919.gui
mv CaMgO_full_tmp_2920.f9 CaMgO_full_2920.f9
mv CaMgO_full_tmp_2920.gui CaMgO_full_2920.gui
mv CaMgO_full_tmp_2921.f9 CaMgO_full_2921.f9
mv CaMgO_full_tmp_2921.gui CaMgO_full_2921.gui
mv CaMgO_full_tmp_2922.f9 CaMgO_full_2922.f9
mv CaMgO_full_tmp_2922.gui CaMgO_full_2922.gui
mv CaMgO_full_tmp_2923.f9 CaMgO_full_2923.f9
mv CaMgO_full_tmp_2923.gui CaMgO_full_2923.gui
mv CaMgO_full_tmp_2924.f9 CaMgO_full_2924.f9
mv CaMgO_full_tmp_2924.gui CaMgO_full_2924.gui
mv CaMgO_full_tmp_2925.f9 CaMgO_full_2925.f9
mv CaMgO_full_tmp_2925.gui CaMgO_full_2925.gui
mv CaMgO_full_tmp_2926.f9 CaMgO_full_2926.f9
mv CaMgO_full_tmp_2926.gui CaMgO_full_2926.gui
mv CaMgO_full_tmp_2927.f9 CaMgO_full_2927.f9
mv CaMgO_full_tmp_2927.gui CaMgO_full_2927.gui
mv CaMgO_full_tmp_2928.f9 CaMgO_full_2928.f9
mv CaMgO_full_tmp_2928.gui CaMgO_full_2928.gui
mv CaMgO_full_tmp_2929.f9 CaMgO_full_2929.f9
mv CaMgO_full_tmp_2929.gui CaMgO_full_2929.gui
mv CaMgO_full_tmp_2930.f9 CaMgO_full_2930.f9
mv CaMgO_full_tmp_2930.gui CaMgO_full_2930.gui
mv CaMgO_full_tmp_2931.f9 CaMgO_full_2931.f9
mv CaMgO_full_tmp_2931.gui CaMgO_full_2931.gui
mv CaMgO_full_tmp_2932.f9 CaMgO_full_2932.f9
mv CaMgO_full_tmp_2932.gui CaMgO_full_2932.gui
mv CaMgO_full_tmp_2933.f9 CaMgO_full_2933.f9
mv CaMgO_full_tmp_2933.gui CaMgO_full_2933.gui
mv CaMgO_full_tmp_2934.f9 CaMgO_full_2934.f9
mv CaMgO_full_tmp_2934.gui CaMgO_full_2934.gui
mv CaMgO_full_tmp_2935.f9 CaMgO_full_2935.f9
mv CaMgO_full_tmp_2935.gui CaMgO_full_2935.gui
mv CaMgO_full_tmp_2936.f9 CaMgO_full_2936.f9
mv CaMgO_full_tmp_2936.gui CaMgO_full_2936.gui
mv CaMgO_full_tmp_2937.f9 CaMgO_full_2937.f9
mv CaMgO_full_tmp_2937.gui CaMgO_full_2937.gui
mv CaMgO_full_tmp_2938.f9 CaMgO_full_2938.f9
mv CaMgO_full_tmp_2938.gui CaMgO_full_2938.gui
mv CaMgO_full_tmp_2939.f9 CaMgO_full_2939.f9
mv CaMgO_full_tmp_2939.gui CaMgO_full_2939.gui
mv CaMgO_full_tmp_2940.f9 CaMgO_full_2940.f9
mv CaMgO_full_tmp_2940.gui CaMgO_full_2940.gui
mv CaMgO_full_tmp_2941.f9 CaMgO_full_2941.f9
mv CaMgO_full_tmp_2941.gui CaMgO_full_2941.gui
mv CaMgO_full_tmp_2942.f9 CaMgO_full_2942.f9
mv CaMgO_full_tmp_2942.gui CaMgO_full_2942.gui
mv CaMgO_full_tmp_2943.f9 CaMgO_full_2943.f9
mv CaMgO_full_tmp_2943.gui CaMgO_full_2943.gui
mv CaMgO_full_tmp_2944.f9 CaMgO_full_2944.f9
mv CaMgO_full_tmp_2944.gui CaMgO_full_2944.gui
mv CaMgO_full_tmp_2945.f9 CaMgO_full_2945.f9
mv CaMgO_full_tmp_2945.gui CaMgO_full_2945.gui
mv CaMgO_full_tmp_2946.f9 CaMgO_full_2946.f9
mv CaMgO_full_tmp_2946.gui CaMgO_full_2946.gui
mv CaMgO_full_tmp_2947.f9 CaMgO_full_2947.f9
mv CaMgO_full_tmp_2947.gui CaMgO_full_2947.gui
mv CaMgO_full_tmp_2948.f9 CaMgO_full_2948.f9
mv CaMgO_full_tmp_2948.gui CaMgO_full_2948.gui
mv CaMgO_full_tmp_2949.f9 CaMgO_full_2949.f9
mv CaMgO_full_tmp_2949.gui CaMgO_full_2949.gui
mv CaMgO_full_tmp_2950.f9 CaMgO_full_2950.f9
mv CaMgO_full_tmp_2950.gui CaMgO_full_2950.gui
mv CaMgO_full_tmp_2951.f9 CaMgO_full_2951.f9
mv CaMgO_full_tmp_2951.gui CaMgO_full_2951.gui
mv CaMgO_full_tmp_2952.f9 CaMgO_full_2952.f9
mv CaMgO_full_tmp_2952.gui CaMgO_full_2952.gui
mv CaMgO_full_tmp_2953.f9 CaMgO_full_2953.f9
mv CaMgO_full_tmp_2953.gui CaMgO_full_2953.gui
mv CaMgO_full_tmp_2954.f9 CaMgO_full_2954.f9
mv CaMgO_full_tmp_2954.gui CaMgO_full_2954.gui
mv CaMgO_full_tmp_2955.f9 CaMgO_full_2955.f9
mv CaMgO_full_tmp_2955.gui CaMgO_full_2955.gui
mv CaMgO_full_tmp_2956.f9 CaMgO_full_2956.f9
mv CaMgO_full_tmp_2956.gui CaMgO_full_2956.gui
mv CaMgO_full_tmp_2957.f9 CaMgO_full_2957.f9
mv CaMgO_full_tmp_2957.gui CaMgO_full_2957.gui
mv CaMgO_full_tmp_2958.f9 CaMgO_full_2958.f9
mv CaMgO_full_tmp_2958.gui CaMgO_full_2958.gui
mv CaMgO_full_tmp_2959.f9 CaMgO_full_2959.f9
mv CaMgO_full_tmp_2959.gui CaMgO_full_2959.gui
mv CaMgO_full_tmp_2960.f9 CaMgO_full_2960.f9
mv CaMgO_full_tmp_2960.gui CaMgO_full_2960.gui
mv CaMgO_full_tmp_2961.f9 CaMgO_full_2961.f9
mv CaMgO_full_tmp_2961.gui CaMgO_full_2961.gui
mv CaMgO_full_tmp_2962.f9 CaMgO_full_2962.f9
mv CaMgO_full_tmp_2962.gui CaMgO_full_2962.gui
mv CaMgO_full_tmp_2963.f9 CaMgO_full_2963.f9
mv CaMgO_full_tmp_2963.gui CaMgO_full_2963.gui
mv CaMgO_full_tmp_2964.f9 CaMgO_full_2964.f9
mv CaMgO_full_tmp_2964.gui CaMgO_full_2964.gui
mv CaMgO_full_tmp_2965.f9 CaMgO_full_2965.f9
mv CaMgO_full_tmp_2965.gui CaMgO_full_2965.gui
mv CaMgO_full_tmp_2966.f9 CaMgO_full_2966.f9
mv CaMgO_full_tmp_2966.gui CaMgO_full_2966.gui
mv CaMgO_full_tmp_2967.f9 CaMgO_full_2967.f9
mv CaMgO_full_tmp_2967.gui CaMgO_full_2967.gui
mv CaMgO_full_tmp_2968.f9 CaMgO_full_2968.f9
mv CaMgO_full_tmp_2968.gui CaMgO_full_2968.gui
mv CaMgO_full_tmp_2969.f9 CaMgO_full_2969.f9
mv CaMgO_full_tmp_2969.gui CaMgO_full_2969.gui
mv CaMgO_full_tmp_2970.f9 CaMgO_full_2970.f9
mv CaMgO_full_tmp_2970.gui CaMgO_full_2970.gui
mv CaMgO_full_tmp_2971.f9 CaMgO_full_2971.f9
mv CaMgO_full_tmp_2971.gui CaMgO_full_2971.gui
mv CaMgO_full_tmp_2972.f9 CaMgO_full_2972.f9
mv CaMgO_full_tmp_2972.gui CaMgO_full_2972.gui
mv CaMgO_full_tmp_2973.f9 CaMgO_full_2973.f9
mv CaMgO_full_tmp_2973.gui CaMgO_full_2973.gui
mv CaMgO_full_tmp_2974.f9 CaMgO_full_2974.f9
mv CaMgO_full_tmp_2974.gui CaMgO_full_2974.gui
mv CaMgO_full_tmp_2975.f9 CaMgO_full_2975.f9
mv CaMgO_full_tmp_2975.gui CaMgO_full_2975.gui
mv CaMgO_full_tmp_2976.f9 CaMgO_full_2976.f9
mv CaMgO_full_tmp_2976.gui CaMgO_full_2976.gui
mv CaMgO_full_tmp_2977.f9 CaMgO_full_2977.f9
mv CaMgO_full_tmp_2977.gui CaMgO_full_2977.gui
mv CaMgO_full_tmp_2978.f9 CaMgO_full_2978.f9
mv CaMgO_full_tmp_2978.gui CaMgO_full_2978.gui
mv CaMgO_full_tmp_2979.f9 CaMgO_full_2979.f9
mv CaMgO_full_tmp_2979.gui CaMgO_full_2979.gui
mv CaMgO_full_tmp_2980.f9 CaMgO_full_2980.f9
mv CaMgO_full_tmp_2980.gui CaMgO_full_2980.gui
mv CaMgO_full_tmp_2981.f9 CaMgO_full_2981.f9
mv CaMgO_full_tmp_2981.gui CaMgO_full_2981.gui
mv CaMgO_full_tmp_2982.f9 CaMgO_full_2982.f9
mv CaMgO_full_tmp_2982.gui CaMgO_full_2982.gui
mv CaMgO_full_tmp_2983.f9 CaMgO_full_2983.f9
mv CaMgO_full_tmp_2983.gui CaMgO_full_2983.gui
mv CaMgO_full_tmp_2984.f9 CaMgO_full_2984.f9
mv CaMgO_full_tmp_2984.gui CaMgO_full_2984.gui
mv CaMgO_full_tmp_2985.f9 CaMgO_full_2985.f9
mv CaMgO_full_tmp_2985.gui CaMgO_full_2985.gui
mv CaMgO_full_tmp_2986.f9 CaMgO_full_2986.f9
mv CaMgO_full_tmp_2986.gui CaMgO_full_2986.gui
mv CaMgO_full_tmp_2987.f9 CaMgO_full_2987.f9
mv CaMgO_full_tmp_2987.gui CaMgO_full_2987.gui
mv CaMgO_full_tmp_2988.f9 CaMgO_full_2988.f9
mv CaMgO_full_tmp_2988.gui CaMgO_full_2988.gui
mv CaMgO_full_tmp_2989.f9 CaMgO_full_2989.f9
mv CaMgO_full_tmp_2989.gui CaMgO_full_2989.gui
mv CaMgO_full_tmp_2990.f9 CaMgO_full_2990.f9
mv CaMgO_full_tmp_2990.gui CaMgO_full_2990.gui
mv CaMgO_full_tmp_2991.f9 CaMgO_full_2991.f9
mv CaMgO_full_tmp_2991.gui CaMgO_full_2991.gui
mv CaMgO_full_tmp_2992.f9 CaMgO_full_2992.f9
mv CaMgO_full_tmp_2992.gui CaMgO_full_2992.gui
mv CaMgO_full_tmp_2993.f9 CaMgO_full_2993.f9
mv CaMgO_full_tmp_2993.gui CaMgO_full_2993.gui
mv CaMgO_full_tmp_2994.f9 CaMgO_full_2994.f9
mv CaMgO_full_tmp_2994.gui CaMgO_full_2994.gui
mv CaMgO_full_tmp_2995.f9 CaMgO_full_2995.f9
mv CaMgO_full_tmp_2995.gui CaMgO_full_2995.gui
mv CaMgO_full_tmp_2996.f9 CaMgO_full_2996.f9
mv CaMgO_full_tmp_2996.gui CaMgO_full_2996.gui
mv CaMgO_full_tmp_2997.f9 CaMgO_full_2997.f9
mv CaMgO_full_tmp_2997.gui CaMgO_full_2997.gui
mv CaMgO_full_tmp_2998.f9 CaMgO_full_2998.f9
mv CaMgO_full_tmp_2998.gui CaMgO_full_2998.gui
mv CaMgO_full_tmp_2999.f9 CaMgO_full_2999.f9
mv CaMgO_full_tmp_2999.gui CaMgO_full_2999.gui
