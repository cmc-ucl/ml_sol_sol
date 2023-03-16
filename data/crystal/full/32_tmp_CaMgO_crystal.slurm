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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3100 CaMgO_full_tmp_3100 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3101 CaMgO_full_tmp_3101 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3102 CaMgO_full_tmp_3102 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3103 CaMgO_full_tmp_3103 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3104 CaMgO_full_tmp_3104 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3105 CaMgO_full_tmp_3105 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3106 CaMgO_full_tmp_3106 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3107 CaMgO_full_tmp_3107 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3108 CaMgO_full_tmp_3108 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3109 CaMgO_full_tmp_3109 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3110 CaMgO_full_tmp_3110 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3111 CaMgO_full_tmp_3111 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3112 CaMgO_full_tmp_3112 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3113 CaMgO_full_tmp_3113 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3114 CaMgO_full_tmp_3114 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3115 CaMgO_full_tmp_3115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3116 CaMgO_full_tmp_3116 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3117 CaMgO_full_tmp_3117 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3118 CaMgO_full_tmp_3118 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3119 CaMgO_full_tmp_3119 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3120 CaMgO_full_tmp_3120 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3121 CaMgO_full_tmp_3121 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3122 CaMgO_full_tmp_3122 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3123 CaMgO_full_tmp_3123 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3124 CaMgO_full_tmp_3124 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3125 CaMgO_full_tmp_3125 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3126 CaMgO_full_tmp_3126 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3127 CaMgO_full_tmp_3127 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3128 CaMgO_full_tmp_3128 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3129 CaMgO_full_tmp_3129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3130 CaMgO_full_tmp_3130 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3131 CaMgO_full_tmp_3131 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3132 CaMgO_full_tmp_3132 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3133 CaMgO_full_tmp_3133 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3134 CaMgO_full_tmp_3134 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3135 CaMgO_full_tmp_3135 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3136 CaMgO_full_tmp_3136 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3137 CaMgO_full_tmp_3137 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3138 CaMgO_full_tmp_3138 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3139 CaMgO_full_tmp_3139 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3140 CaMgO_full_tmp_3140 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3141 CaMgO_full_tmp_3141 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3142 CaMgO_full_tmp_3142 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3143 CaMgO_full_tmp_3143 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3144 CaMgO_full_tmp_3144 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3145 CaMgO_full_tmp_3145 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3146 CaMgO_full_tmp_3146 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3147 CaMgO_full_tmp_3147 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3148 CaMgO_full_tmp_3148 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3149 CaMgO_full_tmp_3149 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3150 CaMgO_full_tmp_3150 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3151 CaMgO_full_tmp_3151 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3152 CaMgO_full_tmp_3152 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3153 CaMgO_full_tmp_3153 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3154 CaMgO_full_tmp_3154 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3155 CaMgO_full_tmp_3155 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3156 CaMgO_full_tmp_3156 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3157 CaMgO_full_tmp_3157 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3158 CaMgO_full_tmp_3158 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3159 CaMgO_full_tmp_3159 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3160 CaMgO_full_tmp_3160 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3161 CaMgO_full_tmp_3161 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3162 CaMgO_full_tmp_3162 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3163 CaMgO_full_tmp_3163 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3164 CaMgO_full_tmp_3164 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3165 CaMgO_full_tmp_3165 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3166 CaMgO_full_tmp_3166 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3167 CaMgO_full_tmp_3167 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3168 CaMgO_full_tmp_3168 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3169 CaMgO_full_tmp_3169 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3170 CaMgO_full_tmp_3170 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3171 CaMgO_full_tmp_3171 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3172 CaMgO_full_tmp_3172 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3173 CaMgO_full_tmp_3173 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3174 CaMgO_full_tmp_3174 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3175 CaMgO_full_tmp_3175 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3176 CaMgO_full_tmp_3176 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3177 CaMgO_full_tmp_3177 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3178 CaMgO_full_tmp_3178 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3179 CaMgO_full_tmp_3179 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3180 CaMgO_full_tmp_3180 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3181 CaMgO_full_tmp_3181 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3182 CaMgO_full_tmp_3182 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3183 CaMgO_full_tmp_3183 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3184 CaMgO_full_tmp_3184 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3185 CaMgO_full_tmp_3185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3186 CaMgO_full_tmp_3186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3187 CaMgO_full_tmp_3187 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3188 CaMgO_full_tmp_3188 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3189 CaMgO_full_tmp_3189 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3190 CaMgO_full_tmp_3190 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3191 CaMgO_full_tmp_3191 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3192 CaMgO_full_tmp_3192 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3193 CaMgO_full_tmp_3193 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3194 CaMgO_full_tmp_3194 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3195 CaMgO_full_tmp_3195 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3196 CaMgO_full_tmp_3196 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3197 CaMgO_full_tmp_3197 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3198 CaMgO_full_tmp_3198 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3199 CaMgO_full_tmp_3199 &

wait
mv CaMgO_full_tmp_3100.f9 CaMgO_full_3100.f9
mv CaMgO_full_tmp_3100.gui CaMgO_full_3100.gui
mv CaMgO_full_tmp_3101.f9 CaMgO_full_3101.f9
mv CaMgO_full_tmp_3101.gui CaMgO_full_3101.gui
mv CaMgO_full_tmp_3102.f9 CaMgO_full_3102.f9
mv CaMgO_full_tmp_3102.gui CaMgO_full_3102.gui
mv CaMgO_full_tmp_3103.f9 CaMgO_full_3103.f9
mv CaMgO_full_tmp_3103.gui CaMgO_full_3103.gui
mv CaMgO_full_tmp_3104.f9 CaMgO_full_3104.f9
mv CaMgO_full_tmp_3104.gui CaMgO_full_3104.gui
mv CaMgO_full_tmp_3105.f9 CaMgO_full_3105.f9
mv CaMgO_full_tmp_3105.gui CaMgO_full_3105.gui
mv CaMgO_full_tmp_3106.f9 CaMgO_full_3106.f9
mv CaMgO_full_tmp_3106.gui CaMgO_full_3106.gui
mv CaMgO_full_tmp_3107.f9 CaMgO_full_3107.f9
mv CaMgO_full_tmp_3107.gui CaMgO_full_3107.gui
mv CaMgO_full_tmp_3108.f9 CaMgO_full_3108.f9
mv CaMgO_full_tmp_3108.gui CaMgO_full_3108.gui
mv CaMgO_full_tmp_3109.f9 CaMgO_full_3109.f9
mv CaMgO_full_tmp_3109.gui CaMgO_full_3109.gui
mv CaMgO_full_tmp_3110.f9 CaMgO_full_3110.f9
mv CaMgO_full_tmp_3110.gui CaMgO_full_3110.gui
mv CaMgO_full_tmp_3111.f9 CaMgO_full_3111.f9
mv CaMgO_full_tmp_3111.gui CaMgO_full_3111.gui
mv CaMgO_full_tmp_3112.f9 CaMgO_full_3112.f9
mv CaMgO_full_tmp_3112.gui CaMgO_full_3112.gui
mv CaMgO_full_tmp_3113.f9 CaMgO_full_3113.f9
mv CaMgO_full_tmp_3113.gui CaMgO_full_3113.gui
mv CaMgO_full_tmp_3114.f9 CaMgO_full_3114.f9
mv CaMgO_full_tmp_3114.gui CaMgO_full_3114.gui
mv CaMgO_full_tmp_3115.f9 CaMgO_full_3115.f9
mv CaMgO_full_tmp_3115.gui CaMgO_full_3115.gui
mv CaMgO_full_tmp_3116.f9 CaMgO_full_3116.f9
mv CaMgO_full_tmp_3116.gui CaMgO_full_3116.gui
mv CaMgO_full_tmp_3117.f9 CaMgO_full_3117.f9
mv CaMgO_full_tmp_3117.gui CaMgO_full_3117.gui
mv CaMgO_full_tmp_3118.f9 CaMgO_full_3118.f9
mv CaMgO_full_tmp_3118.gui CaMgO_full_3118.gui
mv CaMgO_full_tmp_3119.f9 CaMgO_full_3119.f9
mv CaMgO_full_tmp_3119.gui CaMgO_full_3119.gui
mv CaMgO_full_tmp_3120.f9 CaMgO_full_3120.f9
mv CaMgO_full_tmp_3120.gui CaMgO_full_3120.gui
mv CaMgO_full_tmp_3121.f9 CaMgO_full_3121.f9
mv CaMgO_full_tmp_3121.gui CaMgO_full_3121.gui
mv CaMgO_full_tmp_3122.f9 CaMgO_full_3122.f9
mv CaMgO_full_tmp_3122.gui CaMgO_full_3122.gui
mv CaMgO_full_tmp_3123.f9 CaMgO_full_3123.f9
mv CaMgO_full_tmp_3123.gui CaMgO_full_3123.gui
mv CaMgO_full_tmp_3124.f9 CaMgO_full_3124.f9
mv CaMgO_full_tmp_3124.gui CaMgO_full_3124.gui
mv CaMgO_full_tmp_3125.f9 CaMgO_full_3125.f9
mv CaMgO_full_tmp_3125.gui CaMgO_full_3125.gui
mv CaMgO_full_tmp_3126.f9 CaMgO_full_3126.f9
mv CaMgO_full_tmp_3126.gui CaMgO_full_3126.gui
mv CaMgO_full_tmp_3127.f9 CaMgO_full_3127.f9
mv CaMgO_full_tmp_3127.gui CaMgO_full_3127.gui
mv CaMgO_full_tmp_3128.f9 CaMgO_full_3128.f9
mv CaMgO_full_tmp_3128.gui CaMgO_full_3128.gui
mv CaMgO_full_tmp_3129.f9 CaMgO_full_3129.f9
mv CaMgO_full_tmp_3129.gui CaMgO_full_3129.gui
mv CaMgO_full_tmp_3130.f9 CaMgO_full_3130.f9
mv CaMgO_full_tmp_3130.gui CaMgO_full_3130.gui
mv CaMgO_full_tmp_3131.f9 CaMgO_full_3131.f9
mv CaMgO_full_tmp_3131.gui CaMgO_full_3131.gui
mv CaMgO_full_tmp_3132.f9 CaMgO_full_3132.f9
mv CaMgO_full_tmp_3132.gui CaMgO_full_3132.gui
mv CaMgO_full_tmp_3133.f9 CaMgO_full_3133.f9
mv CaMgO_full_tmp_3133.gui CaMgO_full_3133.gui
mv CaMgO_full_tmp_3134.f9 CaMgO_full_3134.f9
mv CaMgO_full_tmp_3134.gui CaMgO_full_3134.gui
mv CaMgO_full_tmp_3135.f9 CaMgO_full_3135.f9
mv CaMgO_full_tmp_3135.gui CaMgO_full_3135.gui
mv CaMgO_full_tmp_3136.f9 CaMgO_full_3136.f9
mv CaMgO_full_tmp_3136.gui CaMgO_full_3136.gui
mv CaMgO_full_tmp_3137.f9 CaMgO_full_3137.f9
mv CaMgO_full_tmp_3137.gui CaMgO_full_3137.gui
mv CaMgO_full_tmp_3138.f9 CaMgO_full_3138.f9
mv CaMgO_full_tmp_3138.gui CaMgO_full_3138.gui
mv CaMgO_full_tmp_3139.f9 CaMgO_full_3139.f9
mv CaMgO_full_tmp_3139.gui CaMgO_full_3139.gui
mv CaMgO_full_tmp_3140.f9 CaMgO_full_3140.f9
mv CaMgO_full_tmp_3140.gui CaMgO_full_3140.gui
mv CaMgO_full_tmp_3141.f9 CaMgO_full_3141.f9
mv CaMgO_full_tmp_3141.gui CaMgO_full_3141.gui
mv CaMgO_full_tmp_3142.f9 CaMgO_full_3142.f9
mv CaMgO_full_tmp_3142.gui CaMgO_full_3142.gui
mv CaMgO_full_tmp_3143.f9 CaMgO_full_3143.f9
mv CaMgO_full_tmp_3143.gui CaMgO_full_3143.gui
mv CaMgO_full_tmp_3144.f9 CaMgO_full_3144.f9
mv CaMgO_full_tmp_3144.gui CaMgO_full_3144.gui
mv CaMgO_full_tmp_3145.f9 CaMgO_full_3145.f9
mv CaMgO_full_tmp_3145.gui CaMgO_full_3145.gui
mv CaMgO_full_tmp_3146.f9 CaMgO_full_3146.f9
mv CaMgO_full_tmp_3146.gui CaMgO_full_3146.gui
mv CaMgO_full_tmp_3147.f9 CaMgO_full_3147.f9
mv CaMgO_full_tmp_3147.gui CaMgO_full_3147.gui
mv CaMgO_full_tmp_3148.f9 CaMgO_full_3148.f9
mv CaMgO_full_tmp_3148.gui CaMgO_full_3148.gui
mv CaMgO_full_tmp_3149.f9 CaMgO_full_3149.f9
mv CaMgO_full_tmp_3149.gui CaMgO_full_3149.gui
mv CaMgO_full_tmp_3150.f9 CaMgO_full_3150.f9
mv CaMgO_full_tmp_3150.gui CaMgO_full_3150.gui
mv CaMgO_full_tmp_3151.f9 CaMgO_full_3151.f9
mv CaMgO_full_tmp_3151.gui CaMgO_full_3151.gui
mv CaMgO_full_tmp_3152.f9 CaMgO_full_3152.f9
mv CaMgO_full_tmp_3152.gui CaMgO_full_3152.gui
mv CaMgO_full_tmp_3153.f9 CaMgO_full_3153.f9
mv CaMgO_full_tmp_3153.gui CaMgO_full_3153.gui
mv CaMgO_full_tmp_3154.f9 CaMgO_full_3154.f9
mv CaMgO_full_tmp_3154.gui CaMgO_full_3154.gui
mv CaMgO_full_tmp_3155.f9 CaMgO_full_3155.f9
mv CaMgO_full_tmp_3155.gui CaMgO_full_3155.gui
mv CaMgO_full_tmp_3156.f9 CaMgO_full_3156.f9
mv CaMgO_full_tmp_3156.gui CaMgO_full_3156.gui
mv CaMgO_full_tmp_3157.f9 CaMgO_full_3157.f9
mv CaMgO_full_tmp_3157.gui CaMgO_full_3157.gui
mv CaMgO_full_tmp_3158.f9 CaMgO_full_3158.f9
mv CaMgO_full_tmp_3158.gui CaMgO_full_3158.gui
mv CaMgO_full_tmp_3159.f9 CaMgO_full_3159.f9
mv CaMgO_full_tmp_3159.gui CaMgO_full_3159.gui
mv CaMgO_full_tmp_3160.f9 CaMgO_full_3160.f9
mv CaMgO_full_tmp_3160.gui CaMgO_full_3160.gui
mv CaMgO_full_tmp_3161.f9 CaMgO_full_3161.f9
mv CaMgO_full_tmp_3161.gui CaMgO_full_3161.gui
mv CaMgO_full_tmp_3162.f9 CaMgO_full_3162.f9
mv CaMgO_full_tmp_3162.gui CaMgO_full_3162.gui
mv CaMgO_full_tmp_3163.f9 CaMgO_full_3163.f9
mv CaMgO_full_tmp_3163.gui CaMgO_full_3163.gui
mv CaMgO_full_tmp_3164.f9 CaMgO_full_3164.f9
mv CaMgO_full_tmp_3164.gui CaMgO_full_3164.gui
mv CaMgO_full_tmp_3165.f9 CaMgO_full_3165.f9
mv CaMgO_full_tmp_3165.gui CaMgO_full_3165.gui
mv CaMgO_full_tmp_3166.f9 CaMgO_full_3166.f9
mv CaMgO_full_tmp_3166.gui CaMgO_full_3166.gui
mv CaMgO_full_tmp_3167.f9 CaMgO_full_3167.f9
mv CaMgO_full_tmp_3167.gui CaMgO_full_3167.gui
mv CaMgO_full_tmp_3168.f9 CaMgO_full_3168.f9
mv CaMgO_full_tmp_3168.gui CaMgO_full_3168.gui
mv CaMgO_full_tmp_3169.f9 CaMgO_full_3169.f9
mv CaMgO_full_tmp_3169.gui CaMgO_full_3169.gui
mv CaMgO_full_tmp_3170.f9 CaMgO_full_3170.f9
mv CaMgO_full_tmp_3170.gui CaMgO_full_3170.gui
mv CaMgO_full_tmp_3171.f9 CaMgO_full_3171.f9
mv CaMgO_full_tmp_3171.gui CaMgO_full_3171.gui
mv CaMgO_full_tmp_3172.f9 CaMgO_full_3172.f9
mv CaMgO_full_tmp_3172.gui CaMgO_full_3172.gui
mv CaMgO_full_tmp_3173.f9 CaMgO_full_3173.f9
mv CaMgO_full_tmp_3173.gui CaMgO_full_3173.gui
mv CaMgO_full_tmp_3174.f9 CaMgO_full_3174.f9
mv CaMgO_full_tmp_3174.gui CaMgO_full_3174.gui
mv CaMgO_full_tmp_3175.f9 CaMgO_full_3175.f9
mv CaMgO_full_tmp_3175.gui CaMgO_full_3175.gui
mv CaMgO_full_tmp_3176.f9 CaMgO_full_3176.f9
mv CaMgO_full_tmp_3176.gui CaMgO_full_3176.gui
mv CaMgO_full_tmp_3177.f9 CaMgO_full_3177.f9
mv CaMgO_full_tmp_3177.gui CaMgO_full_3177.gui
mv CaMgO_full_tmp_3178.f9 CaMgO_full_3178.f9
mv CaMgO_full_tmp_3178.gui CaMgO_full_3178.gui
mv CaMgO_full_tmp_3179.f9 CaMgO_full_3179.f9
mv CaMgO_full_tmp_3179.gui CaMgO_full_3179.gui
mv CaMgO_full_tmp_3180.f9 CaMgO_full_3180.f9
mv CaMgO_full_tmp_3180.gui CaMgO_full_3180.gui
mv CaMgO_full_tmp_3181.f9 CaMgO_full_3181.f9
mv CaMgO_full_tmp_3181.gui CaMgO_full_3181.gui
mv CaMgO_full_tmp_3182.f9 CaMgO_full_3182.f9
mv CaMgO_full_tmp_3182.gui CaMgO_full_3182.gui
mv CaMgO_full_tmp_3183.f9 CaMgO_full_3183.f9
mv CaMgO_full_tmp_3183.gui CaMgO_full_3183.gui
mv CaMgO_full_tmp_3184.f9 CaMgO_full_3184.f9
mv CaMgO_full_tmp_3184.gui CaMgO_full_3184.gui
mv CaMgO_full_tmp_3185.f9 CaMgO_full_3185.f9
mv CaMgO_full_tmp_3185.gui CaMgO_full_3185.gui
mv CaMgO_full_tmp_3186.f9 CaMgO_full_3186.f9
mv CaMgO_full_tmp_3186.gui CaMgO_full_3186.gui
mv CaMgO_full_tmp_3187.f9 CaMgO_full_3187.f9
mv CaMgO_full_tmp_3187.gui CaMgO_full_3187.gui
mv CaMgO_full_tmp_3188.f9 CaMgO_full_3188.f9
mv CaMgO_full_tmp_3188.gui CaMgO_full_3188.gui
mv CaMgO_full_tmp_3189.f9 CaMgO_full_3189.f9
mv CaMgO_full_tmp_3189.gui CaMgO_full_3189.gui
mv CaMgO_full_tmp_3190.f9 CaMgO_full_3190.f9
mv CaMgO_full_tmp_3190.gui CaMgO_full_3190.gui
mv CaMgO_full_tmp_3191.f9 CaMgO_full_3191.f9
mv CaMgO_full_tmp_3191.gui CaMgO_full_3191.gui
mv CaMgO_full_tmp_3192.f9 CaMgO_full_3192.f9
mv CaMgO_full_tmp_3192.gui CaMgO_full_3192.gui
mv CaMgO_full_tmp_3193.f9 CaMgO_full_3193.f9
mv CaMgO_full_tmp_3193.gui CaMgO_full_3193.gui
mv CaMgO_full_tmp_3194.f9 CaMgO_full_3194.f9
mv CaMgO_full_tmp_3194.gui CaMgO_full_3194.gui
mv CaMgO_full_tmp_3195.f9 CaMgO_full_3195.f9
mv CaMgO_full_tmp_3195.gui CaMgO_full_3195.gui
mv CaMgO_full_tmp_3196.f9 CaMgO_full_3196.f9
mv CaMgO_full_tmp_3196.gui CaMgO_full_3196.gui
mv CaMgO_full_tmp_3197.f9 CaMgO_full_3197.f9
mv CaMgO_full_tmp_3197.gui CaMgO_full_3197.gui
mv CaMgO_full_tmp_3198.f9 CaMgO_full_3198.f9
mv CaMgO_full_tmp_3198.gui CaMgO_full_3198.gui
mv CaMgO_full_tmp_3199.f9 CaMgO_full_3199.f9
mv CaMgO_full_tmp_3199.gui CaMgO_full_3199.gui
