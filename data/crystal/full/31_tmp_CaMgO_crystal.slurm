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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3000 CaMgO_full_tmp_3000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3001 CaMgO_full_tmp_3001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3002 CaMgO_full_tmp_3002 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3003 CaMgO_full_tmp_3003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3004 CaMgO_full_tmp_3004 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3005 CaMgO_full_tmp_3005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3006 CaMgO_full_tmp_3006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3007 CaMgO_full_tmp_3007 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3008 CaMgO_full_tmp_3008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3009 CaMgO_full_tmp_3009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3010 CaMgO_full_tmp_3010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3011 CaMgO_full_tmp_3011 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3012 CaMgO_full_tmp_3012 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3013 CaMgO_full_tmp_3013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3014 CaMgO_full_tmp_3014 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3015 CaMgO_full_tmp_3015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3016 CaMgO_full_tmp_3016 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3017 CaMgO_full_tmp_3017 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3018 CaMgO_full_tmp_3018 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3019 CaMgO_full_tmp_3019 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3020 CaMgO_full_tmp_3020 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3021 CaMgO_full_tmp_3021 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3022 CaMgO_full_tmp_3022 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3023 CaMgO_full_tmp_3023 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3024 CaMgO_full_tmp_3024 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3025 CaMgO_full_tmp_3025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3026 CaMgO_full_tmp_3026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3027 CaMgO_full_tmp_3027 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3028 CaMgO_full_tmp_3028 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3029 CaMgO_full_tmp_3029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3030 CaMgO_full_tmp_3030 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3031 CaMgO_full_tmp_3031 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3032 CaMgO_full_tmp_3032 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3033 CaMgO_full_tmp_3033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3034 CaMgO_full_tmp_3034 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3035 CaMgO_full_tmp_3035 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3036 CaMgO_full_tmp_3036 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3037 CaMgO_full_tmp_3037 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3038 CaMgO_full_tmp_3038 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3039 CaMgO_full_tmp_3039 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3040 CaMgO_full_tmp_3040 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3041 CaMgO_full_tmp_3041 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3042 CaMgO_full_tmp_3042 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3043 CaMgO_full_tmp_3043 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3044 CaMgO_full_tmp_3044 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3045 CaMgO_full_tmp_3045 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3046 CaMgO_full_tmp_3046 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3047 CaMgO_full_tmp_3047 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3048 CaMgO_full_tmp_3048 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3049 CaMgO_full_tmp_3049 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3050 CaMgO_full_tmp_3050 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3051 CaMgO_full_tmp_3051 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3052 CaMgO_full_tmp_3052 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3053 CaMgO_full_tmp_3053 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3054 CaMgO_full_tmp_3054 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3055 CaMgO_full_tmp_3055 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3056 CaMgO_full_tmp_3056 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3057 CaMgO_full_tmp_3057 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3058 CaMgO_full_tmp_3058 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3059 CaMgO_full_tmp_3059 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3060 CaMgO_full_tmp_3060 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3061 CaMgO_full_tmp_3061 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3062 CaMgO_full_tmp_3062 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3063 CaMgO_full_tmp_3063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3064 CaMgO_full_tmp_3064 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3065 CaMgO_full_tmp_3065 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3066 CaMgO_full_tmp_3066 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3067 CaMgO_full_tmp_3067 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3068 CaMgO_full_tmp_3068 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3069 CaMgO_full_tmp_3069 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3070 CaMgO_full_tmp_3070 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3071 CaMgO_full_tmp_3071 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3072 CaMgO_full_tmp_3072 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3073 CaMgO_full_tmp_3073 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3074 CaMgO_full_tmp_3074 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3075 CaMgO_full_tmp_3075 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3076 CaMgO_full_tmp_3076 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3077 CaMgO_full_tmp_3077 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3078 CaMgO_full_tmp_3078 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3079 CaMgO_full_tmp_3079 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3080 CaMgO_full_tmp_3080 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3081 CaMgO_full_tmp_3081 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3082 CaMgO_full_tmp_3082 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3083 CaMgO_full_tmp_3083 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3084 CaMgO_full_tmp_3084 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3085 CaMgO_full_tmp_3085 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3086 CaMgO_full_tmp_3086 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3087 CaMgO_full_tmp_3087 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3088 CaMgO_full_tmp_3088 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3089 CaMgO_full_tmp_3089 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3090 CaMgO_full_tmp_3090 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3091 CaMgO_full_tmp_3091 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3092 CaMgO_full_tmp_3092 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3093 CaMgO_full_tmp_3093 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3094 CaMgO_full_tmp_3094 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3095 CaMgO_full_tmp_3095 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3096 CaMgO_full_tmp_3096 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3097 CaMgO_full_tmp_3097 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3098 CaMgO_full_tmp_3098 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3099 CaMgO_full_tmp_3099 &

wait
mv CaMgO_full_tmp_3000.f9 CaMgO_full_3000.f9
mv CaMgO_full_tmp_3000.gui CaMgO_full_3000.gui
mv CaMgO_full_tmp_3001.f9 CaMgO_full_3001.f9
mv CaMgO_full_tmp_3001.gui CaMgO_full_3001.gui
mv CaMgO_full_tmp_3002.f9 CaMgO_full_3002.f9
mv CaMgO_full_tmp_3002.gui CaMgO_full_3002.gui
mv CaMgO_full_tmp_3003.f9 CaMgO_full_3003.f9
mv CaMgO_full_tmp_3003.gui CaMgO_full_3003.gui
mv CaMgO_full_tmp_3004.f9 CaMgO_full_3004.f9
mv CaMgO_full_tmp_3004.gui CaMgO_full_3004.gui
mv CaMgO_full_tmp_3005.f9 CaMgO_full_3005.f9
mv CaMgO_full_tmp_3005.gui CaMgO_full_3005.gui
mv CaMgO_full_tmp_3006.f9 CaMgO_full_3006.f9
mv CaMgO_full_tmp_3006.gui CaMgO_full_3006.gui
mv CaMgO_full_tmp_3007.f9 CaMgO_full_3007.f9
mv CaMgO_full_tmp_3007.gui CaMgO_full_3007.gui
mv CaMgO_full_tmp_3008.f9 CaMgO_full_3008.f9
mv CaMgO_full_tmp_3008.gui CaMgO_full_3008.gui
mv CaMgO_full_tmp_3009.f9 CaMgO_full_3009.f9
mv CaMgO_full_tmp_3009.gui CaMgO_full_3009.gui
mv CaMgO_full_tmp_3010.f9 CaMgO_full_3010.f9
mv CaMgO_full_tmp_3010.gui CaMgO_full_3010.gui
mv CaMgO_full_tmp_3011.f9 CaMgO_full_3011.f9
mv CaMgO_full_tmp_3011.gui CaMgO_full_3011.gui
mv CaMgO_full_tmp_3012.f9 CaMgO_full_3012.f9
mv CaMgO_full_tmp_3012.gui CaMgO_full_3012.gui
mv CaMgO_full_tmp_3013.f9 CaMgO_full_3013.f9
mv CaMgO_full_tmp_3013.gui CaMgO_full_3013.gui
mv CaMgO_full_tmp_3014.f9 CaMgO_full_3014.f9
mv CaMgO_full_tmp_3014.gui CaMgO_full_3014.gui
mv CaMgO_full_tmp_3015.f9 CaMgO_full_3015.f9
mv CaMgO_full_tmp_3015.gui CaMgO_full_3015.gui
mv CaMgO_full_tmp_3016.f9 CaMgO_full_3016.f9
mv CaMgO_full_tmp_3016.gui CaMgO_full_3016.gui
mv CaMgO_full_tmp_3017.f9 CaMgO_full_3017.f9
mv CaMgO_full_tmp_3017.gui CaMgO_full_3017.gui
mv CaMgO_full_tmp_3018.f9 CaMgO_full_3018.f9
mv CaMgO_full_tmp_3018.gui CaMgO_full_3018.gui
mv CaMgO_full_tmp_3019.f9 CaMgO_full_3019.f9
mv CaMgO_full_tmp_3019.gui CaMgO_full_3019.gui
mv CaMgO_full_tmp_3020.f9 CaMgO_full_3020.f9
mv CaMgO_full_tmp_3020.gui CaMgO_full_3020.gui
mv CaMgO_full_tmp_3021.f9 CaMgO_full_3021.f9
mv CaMgO_full_tmp_3021.gui CaMgO_full_3021.gui
mv CaMgO_full_tmp_3022.f9 CaMgO_full_3022.f9
mv CaMgO_full_tmp_3022.gui CaMgO_full_3022.gui
mv CaMgO_full_tmp_3023.f9 CaMgO_full_3023.f9
mv CaMgO_full_tmp_3023.gui CaMgO_full_3023.gui
mv CaMgO_full_tmp_3024.f9 CaMgO_full_3024.f9
mv CaMgO_full_tmp_3024.gui CaMgO_full_3024.gui
mv CaMgO_full_tmp_3025.f9 CaMgO_full_3025.f9
mv CaMgO_full_tmp_3025.gui CaMgO_full_3025.gui
mv CaMgO_full_tmp_3026.f9 CaMgO_full_3026.f9
mv CaMgO_full_tmp_3026.gui CaMgO_full_3026.gui
mv CaMgO_full_tmp_3027.f9 CaMgO_full_3027.f9
mv CaMgO_full_tmp_3027.gui CaMgO_full_3027.gui
mv CaMgO_full_tmp_3028.f9 CaMgO_full_3028.f9
mv CaMgO_full_tmp_3028.gui CaMgO_full_3028.gui
mv CaMgO_full_tmp_3029.f9 CaMgO_full_3029.f9
mv CaMgO_full_tmp_3029.gui CaMgO_full_3029.gui
mv CaMgO_full_tmp_3030.f9 CaMgO_full_3030.f9
mv CaMgO_full_tmp_3030.gui CaMgO_full_3030.gui
mv CaMgO_full_tmp_3031.f9 CaMgO_full_3031.f9
mv CaMgO_full_tmp_3031.gui CaMgO_full_3031.gui
mv CaMgO_full_tmp_3032.f9 CaMgO_full_3032.f9
mv CaMgO_full_tmp_3032.gui CaMgO_full_3032.gui
mv CaMgO_full_tmp_3033.f9 CaMgO_full_3033.f9
mv CaMgO_full_tmp_3033.gui CaMgO_full_3033.gui
mv CaMgO_full_tmp_3034.f9 CaMgO_full_3034.f9
mv CaMgO_full_tmp_3034.gui CaMgO_full_3034.gui
mv CaMgO_full_tmp_3035.f9 CaMgO_full_3035.f9
mv CaMgO_full_tmp_3035.gui CaMgO_full_3035.gui
mv CaMgO_full_tmp_3036.f9 CaMgO_full_3036.f9
mv CaMgO_full_tmp_3036.gui CaMgO_full_3036.gui
mv CaMgO_full_tmp_3037.f9 CaMgO_full_3037.f9
mv CaMgO_full_tmp_3037.gui CaMgO_full_3037.gui
mv CaMgO_full_tmp_3038.f9 CaMgO_full_3038.f9
mv CaMgO_full_tmp_3038.gui CaMgO_full_3038.gui
mv CaMgO_full_tmp_3039.f9 CaMgO_full_3039.f9
mv CaMgO_full_tmp_3039.gui CaMgO_full_3039.gui
mv CaMgO_full_tmp_3040.f9 CaMgO_full_3040.f9
mv CaMgO_full_tmp_3040.gui CaMgO_full_3040.gui
mv CaMgO_full_tmp_3041.f9 CaMgO_full_3041.f9
mv CaMgO_full_tmp_3041.gui CaMgO_full_3041.gui
mv CaMgO_full_tmp_3042.f9 CaMgO_full_3042.f9
mv CaMgO_full_tmp_3042.gui CaMgO_full_3042.gui
mv CaMgO_full_tmp_3043.f9 CaMgO_full_3043.f9
mv CaMgO_full_tmp_3043.gui CaMgO_full_3043.gui
mv CaMgO_full_tmp_3044.f9 CaMgO_full_3044.f9
mv CaMgO_full_tmp_3044.gui CaMgO_full_3044.gui
mv CaMgO_full_tmp_3045.f9 CaMgO_full_3045.f9
mv CaMgO_full_tmp_3045.gui CaMgO_full_3045.gui
mv CaMgO_full_tmp_3046.f9 CaMgO_full_3046.f9
mv CaMgO_full_tmp_3046.gui CaMgO_full_3046.gui
mv CaMgO_full_tmp_3047.f9 CaMgO_full_3047.f9
mv CaMgO_full_tmp_3047.gui CaMgO_full_3047.gui
mv CaMgO_full_tmp_3048.f9 CaMgO_full_3048.f9
mv CaMgO_full_tmp_3048.gui CaMgO_full_3048.gui
mv CaMgO_full_tmp_3049.f9 CaMgO_full_3049.f9
mv CaMgO_full_tmp_3049.gui CaMgO_full_3049.gui
mv CaMgO_full_tmp_3050.f9 CaMgO_full_3050.f9
mv CaMgO_full_tmp_3050.gui CaMgO_full_3050.gui
mv CaMgO_full_tmp_3051.f9 CaMgO_full_3051.f9
mv CaMgO_full_tmp_3051.gui CaMgO_full_3051.gui
mv CaMgO_full_tmp_3052.f9 CaMgO_full_3052.f9
mv CaMgO_full_tmp_3052.gui CaMgO_full_3052.gui
mv CaMgO_full_tmp_3053.f9 CaMgO_full_3053.f9
mv CaMgO_full_tmp_3053.gui CaMgO_full_3053.gui
mv CaMgO_full_tmp_3054.f9 CaMgO_full_3054.f9
mv CaMgO_full_tmp_3054.gui CaMgO_full_3054.gui
mv CaMgO_full_tmp_3055.f9 CaMgO_full_3055.f9
mv CaMgO_full_tmp_3055.gui CaMgO_full_3055.gui
mv CaMgO_full_tmp_3056.f9 CaMgO_full_3056.f9
mv CaMgO_full_tmp_3056.gui CaMgO_full_3056.gui
mv CaMgO_full_tmp_3057.f9 CaMgO_full_3057.f9
mv CaMgO_full_tmp_3057.gui CaMgO_full_3057.gui
mv CaMgO_full_tmp_3058.f9 CaMgO_full_3058.f9
mv CaMgO_full_tmp_3058.gui CaMgO_full_3058.gui
mv CaMgO_full_tmp_3059.f9 CaMgO_full_3059.f9
mv CaMgO_full_tmp_3059.gui CaMgO_full_3059.gui
mv CaMgO_full_tmp_3060.f9 CaMgO_full_3060.f9
mv CaMgO_full_tmp_3060.gui CaMgO_full_3060.gui
mv CaMgO_full_tmp_3061.f9 CaMgO_full_3061.f9
mv CaMgO_full_tmp_3061.gui CaMgO_full_3061.gui
mv CaMgO_full_tmp_3062.f9 CaMgO_full_3062.f9
mv CaMgO_full_tmp_3062.gui CaMgO_full_3062.gui
mv CaMgO_full_tmp_3063.f9 CaMgO_full_3063.f9
mv CaMgO_full_tmp_3063.gui CaMgO_full_3063.gui
mv CaMgO_full_tmp_3064.f9 CaMgO_full_3064.f9
mv CaMgO_full_tmp_3064.gui CaMgO_full_3064.gui
mv CaMgO_full_tmp_3065.f9 CaMgO_full_3065.f9
mv CaMgO_full_tmp_3065.gui CaMgO_full_3065.gui
mv CaMgO_full_tmp_3066.f9 CaMgO_full_3066.f9
mv CaMgO_full_tmp_3066.gui CaMgO_full_3066.gui
mv CaMgO_full_tmp_3067.f9 CaMgO_full_3067.f9
mv CaMgO_full_tmp_3067.gui CaMgO_full_3067.gui
mv CaMgO_full_tmp_3068.f9 CaMgO_full_3068.f9
mv CaMgO_full_tmp_3068.gui CaMgO_full_3068.gui
mv CaMgO_full_tmp_3069.f9 CaMgO_full_3069.f9
mv CaMgO_full_tmp_3069.gui CaMgO_full_3069.gui
mv CaMgO_full_tmp_3070.f9 CaMgO_full_3070.f9
mv CaMgO_full_tmp_3070.gui CaMgO_full_3070.gui
mv CaMgO_full_tmp_3071.f9 CaMgO_full_3071.f9
mv CaMgO_full_tmp_3071.gui CaMgO_full_3071.gui
mv CaMgO_full_tmp_3072.f9 CaMgO_full_3072.f9
mv CaMgO_full_tmp_3072.gui CaMgO_full_3072.gui
mv CaMgO_full_tmp_3073.f9 CaMgO_full_3073.f9
mv CaMgO_full_tmp_3073.gui CaMgO_full_3073.gui
mv CaMgO_full_tmp_3074.f9 CaMgO_full_3074.f9
mv CaMgO_full_tmp_3074.gui CaMgO_full_3074.gui
mv CaMgO_full_tmp_3075.f9 CaMgO_full_3075.f9
mv CaMgO_full_tmp_3075.gui CaMgO_full_3075.gui
mv CaMgO_full_tmp_3076.f9 CaMgO_full_3076.f9
mv CaMgO_full_tmp_3076.gui CaMgO_full_3076.gui
mv CaMgO_full_tmp_3077.f9 CaMgO_full_3077.f9
mv CaMgO_full_tmp_3077.gui CaMgO_full_3077.gui
mv CaMgO_full_tmp_3078.f9 CaMgO_full_3078.f9
mv CaMgO_full_tmp_3078.gui CaMgO_full_3078.gui
mv CaMgO_full_tmp_3079.f9 CaMgO_full_3079.f9
mv CaMgO_full_tmp_3079.gui CaMgO_full_3079.gui
mv CaMgO_full_tmp_3080.f9 CaMgO_full_3080.f9
mv CaMgO_full_tmp_3080.gui CaMgO_full_3080.gui
mv CaMgO_full_tmp_3081.f9 CaMgO_full_3081.f9
mv CaMgO_full_tmp_3081.gui CaMgO_full_3081.gui
mv CaMgO_full_tmp_3082.f9 CaMgO_full_3082.f9
mv CaMgO_full_tmp_3082.gui CaMgO_full_3082.gui
mv CaMgO_full_tmp_3083.f9 CaMgO_full_3083.f9
mv CaMgO_full_tmp_3083.gui CaMgO_full_3083.gui
mv CaMgO_full_tmp_3084.f9 CaMgO_full_3084.f9
mv CaMgO_full_tmp_3084.gui CaMgO_full_3084.gui
mv CaMgO_full_tmp_3085.f9 CaMgO_full_3085.f9
mv CaMgO_full_tmp_3085.gui CaMgO_full_3085.gui
mv CaMgO_full_tmp_3086.f9 CaMgO_full_3086.f9
mv CaMgO_full_tmp_3086.gui CaMgO_full_3086.gui
mv CaMgO_full_tmp_3087.f9 CaMgO_full_3087.f9
mv CaMgO_full_tmp_3087.gui CaMgO_full_3087.gui
mv CaMgO_full_tmp_3088.f9 CaMgO_full_3088.f9
mv CaMgO_full_tmp_3088.gui CaMgO_full_3088.gui
mv CaMgO_full_tmp_3089.f9 CaMgO_full_3089.f9
mv CaMgO_full_tmp_3089.gui CaMgO_full_3089.gui
mv CaMgO_full_tmp_3090.f9 CaMgO_full_3090.f9
mv CaMgO_full_tmp_3090.gui CaMgO_full_3090.gui
mv CaMgO_full_tmp_3091.f9 CaMgO_full_3091.f9
mv CaMgO_full_tmp_3091.gui CaMgO_full_3091.gui
mv CaMgO_full_tmp_3092.f9 CaMgO_full_3092.f9
mv CaMgO_full_tmp_3092.gui CaMgO_full_3092.gui
mv CaMgO_full_tmp_3093.f9 CaMgO_full_3093.f9
mv CaMgO_full_tmp_3093.gui CaMgO_full_3093.gui
mv CaMgO_full_tmp_3094.f9 CaMgO_full_3094.f9
mv CaMgO_full_tmp_3094.gui CaMgO_full_3094.gui
mv CaMgO_full_tmp_3095.f9 CaMgO_full_3095.f9
mv CaMgO_full_tmp_3095.gui CaMgO_full_3095.gui
mv CaMgO_full_tmp_3096.f9 CaMgO_full_3096.f9
mv CaMgO_full_tmp_3096.gui CaMgO_full_3096.gui
mv CaMgO_full_tmp_3097.f9 CaMgO_full_3097.f9
mv CaMgO_full_tmp_3097.gui CaMgO_full_3097.gui
mv CaMgO_full_tmp_3098.f9 CaMgO_full_3098.f9
mv CaMgO_full_tmp_3098.gui CaMgO_full_3098.gui
mv CaMgO_full_tmp_3099.f9 CaMgO_full_3099.f9
mv CaMgO_full_tmp_3099.gui CaMgO_full_3099.gui
