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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3200 CaMgO_full_tmp_3200 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3201 CaMgO_full_tmp_3201 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3202 CaMgO_full_tmp_3202 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3203 CaMgO_full_tmp_3203 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3204 CaMgO_full_tmp_3204 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3205 CaMgO_full_tmp_3205 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3206 CaMgO_full_tmp_3206 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3207 CaMgO_full_tmp_3207 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3208 CaMgO_full_tmp_3208 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3209 CaMgO_full_tmp_3209 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3210 CaMgO_full_tmp_3210 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3211 CaMgO_full_tmp_3211 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3212 CaMgO_full_tmp_3212 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3213 CaMgO_full_tmp_3213 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3214 CaMgO_full_tmp_3214 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3215 CaMgO_full_tmp_3215 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3216 CaMgO_full_tmp_3216 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3217 CaMgO_full_tmp_3217 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3218 CaMgO_full_tmp_3218 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3219 CaMgO_full_tmp_3219 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3220 CaMgO_full_tmp_3220 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3221 CaMgO_full_tmp_3221 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3222 CaMgO_full_tmp_3222 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3223 CaMgO_full_tmp_3223 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3224 CaMgO_full_tmp_3224 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3225 CaMgO_full_tmp_3225 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3226 CaMgO_full_tmp_3226 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3227 CaMgO_full_tmp_3227 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3228 CaMgO_full_tmp_3228 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3229 CaMgO_full_tmp_3229 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3230 CaMgO_full_tmp_3230 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3231 CaMgO_full_tmp_3231 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3232 CaMgO_full_tmp_3232 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3233 CaMgO_full_tmp_3233 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3234 CaMgO_full_tmp_3234 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3235 CaMgO_full_tmp_3235 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3236 CaMgO_full_tmp_3236 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3237 CaMgO_full_tmp_3237 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3238 CaMgO_full_tmp_3238 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3239 CaMgO_full_tmp_3239 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3240 CaMgO_full_tmp_3240 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3241 CaMgO_full_tmp_3241 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3242 CaMgO_full_tmp_3242 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3243 CaMgO_full_tmp_3243 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3244 CaMgO_full_tmp_3244 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3245 CaMgO_full_tmp_3245 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3246 CaMgO_full_tmp_3246 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3247 CaMgO_full_tmp_3247 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3248 CaMgO_full_tmp_3248 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3249 CaMgO_full_tmp_3249 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3250 CaMgO_full_tmp_3250 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3251 CaMgO_full_tmp_3251 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3252 CaMgO_full_tmp_3252 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3253 CaMgO_full_tmp_3253 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3254 CaMgO_full_tmp_3254 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3255 CaMgO_full_tmp_3255 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3256 CaMgO_full_tmp_3256 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3257 CaMgO_full_tmp_3257 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3258 CaMgO_full_tmp_3258 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3259 CaMgO_full_tmp_3259 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3260 CaMgO_full_tmp_3260 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3261 CaMgO_full_tmp_3261 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3262 CaMgO_full_tmp_3262 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3263 CaMgO_full_tmp_3263 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3264 CaMgO_full_tmp_3264 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3265 CaMgO_full_tmp_3265 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3266 CaMgO_full_tmp_3266 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3267 CaMgO_full_tmp_3267 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3268 CaMgO_full_tmp_3268 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3269 CaMgO_full_tmp_3269 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3270 CaMgO_full_tmp_3270 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3271 CaMgO_full_tmp_3271 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3272 CaMgO_full_tmp_3272 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3273 CaMgO_full_tmp_3273 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3274 CaMgO_full_tmp_3274 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3275 CaMgO_full_tmp_3275 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3276 CaMgO_full_tmp_3276 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3277 CaMgO_full_tmp_3277 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3278 CaMgO_full_tmp_3278 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3279 CaMgO_full_tmp_3279 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3280 CaMgO_full_tmp_3280 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3281 CaMgO_full_tmp_3281 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3282 CaMgO_full_tmp_3282 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3283 CaMgO_full_tmp_3283 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3284 CaMgO_full_tmp_3284 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3285 CaMgO_full_tmp_3285 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3286 CaMgO_full_tmp_3286 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3287 CaMgO_full_tmp_3287 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3288 CaMgO_full_tmp_3288 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3289 CaMgO_full_tmp_3289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3290 CaMgO_full_tmp_3290 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3291 CaMgO_full_tmp_3291 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3292 CaMgO_full_tmp_3292 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3293 CaMgO_full_tmp_3293 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3294 CaMgO_full_tmp_3294 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3295 CaMgO_full_tmp_3295 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3296 CaMgO_full_tmp_3296 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3297 CaMgO_full_tmp_3297 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3298 CaMgO_full_tmp_3298 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3299 CaMgO_full_tmp_3299 &

wait
mv CaMgO_full_tmp_3200.f9 CaMgO_full_3200.f9
mv CaMgO_full_tmp_3200.gui CaMgO_full_3200.gui
mv CaMgO_full_tmp_3201.f9 CaMgO_full_3201.f9
mv CaMgO_full_tmp_3201.gui CaMgO_full_3201.gui
mv CaMgO_full_tmp_3202.f9 CaMgO_full_3202.f9
mv CaMgO_full_tmp_3202.gui CaMgO_full_3202.gui
mv CaMgO_full_tmp_3203.f9 CaMgO_full_3203.f9
mv CaMgO_full_tmp_3203.gui CaMgO_full_3203.gui
mv CaMgO_full_tmp_3204.f9 CaMgO_full_3204.f9
mv CaMgO_full_tmp_3204.gui CaMgO_full_3204.gui
mv CaMgO_full_tmp_3205.f9 CaMgO_full_3205.f9
mv CaMgO_full_tmp_3205.gui CaMgO_full_3205.gui
mv CaMgO_full_tmp_3206.f9 CaMgO_full_3206.f9
mv CaMgO_full_tmp_3206.gui CaMgO_full_3206.gui
mv CaMgO_full_tmp_3207.f9 CaMgO_full_3207.f9
mv CaMgO_full_tmp_3207.gui CaMgO_full_3207.gui
mv CaMgO_full_tmp_3208.f9 CaMgO_full_3208.f9
mv CaMgO_full_tmp_3208.gui CaMgO_full_3208.gui
mv CaMgO_full_tmp_3209.f9 CaMgO_full_3209.f9
mv CaMgO_full_tmp_3209.gui CaMgO_full_3209.gui
mv CaMgO_full_tmp_3210.f9 CaMgO_full_3210.f9
mv CaMgO_full_tmp_3210.gui CaMgO_full_3210.gui
mv CaMgO_full_tmp_3211.f9 CaMgO_full_3211.f9
mv CaMgO_full_tmp_3211.gui CaMgO_full_3211.gui
mv CaMgO_full_tmp_3212.f9 CaMgO_full_3212.f9
mv CaMgO_full_tmp_3212.gui CaMgO_full_3212.gui
mv CaMgO_full_tmp_3213.f9 CaMgO_full_3213.f9
mv CaMgO_full_tmp_3213.gui CaMgO_full_3213.gui
mv CaMgO_full_tmp_3214.f9 CaMgO_full_3214.f9
mv CaMgO_full_tmp_3214.gui CaMgO_full_3214.gui
mv CaMgO_full_tmp_3215.f9 CaMgO_full_3215.f9
mv CaMgO_full_tmp_3215.gui CaMgO_full_3215.gui
mv CaMgO_full_tmp_3216.f9 CaMgO_full_3216.f9
mv CaMgO_full_tmp_3216.gui CaMgO_full_3216.gui
mv CaMgO_full_tmp_3217.f9 CaMgO_full_3217.f9
mv CaMgO_full_tmp_3217.gui CaMgO_full_3217.gui
mv CaMgO_full_tmp_3218.f9 CaMgO_full_3218.f9
mv CaMgO_full_tmp_3218.gui CaMgO_full_3218.gui
mv CaMgO_full_tmp_3219.f9 CaMgO_full_3219.f9
mv CaMgO_full_tmp_3219.gui CaMgO_full_3219.gui
mv CaMgO_full_tmp_3220.f9 CaMgO_full_3220.f9
mv CaMgO_full_tmp_3220.gui CaMgO_full_3220.gui
mv CaMgO_full_tmp_3221.f9 CaMgO_full_3221.f9
mv CaMgO_full_tmp_3221.gui CaMgO_full_3221.gui
mv CaMgO_full_tmp_3222.f9 CaMgO_full_3222.f9
mv CaMgO_full_tmp_3222.gui CaMgO_full_3222.gui
mv CaMgO_full_tmp_3223.f9 CaMgO_full_3223.f9
mv CaMgO_full_tmp_3223.gui CaMgO_full_3223.gui
mv CaMgO_full_tmp_3224.f9 CaMgO_full_3224.f9
mv CaMgO_full_tmp_3224.gui CaMgO_full_3224.gui
mv CaMgO_full_tmp_3225.f9 CaMgO_full_3225.f9
mv CaMgO_full_tmp_3225.gui CaMgO_full_3225.gui
mv CaMgO_full_tmp_3226.f9 CaMgO_full_3226.f9
mv CaMgO_full_tmp_3226.gui CaMgO_full_3226.gui
mv CaMgO_full_tmp_3227.f9 CaMgO_full_3227.f9
mv CaMgO_full_tmp_3227.gui CaMgO_full_3227.gui
mv CaMgO_full_tmp_3228.f9 CaMgO_full_3228.f9
mv CaMgO_full_tmp_3228.gui CaMgO_full_3228.gui
mv CaMgO_full_tmp_3229.f9 CaMgO_full_3229.f9
mv CaMgO_full_tmp_3229.gui CaMgO_full_3229.gui
mv CaMgO_full_tmp_3230.f9 CaMgO_full_3230.f9
mv CaMgO_full_tmp_3230.gui CaMgO_full_3230.gui
mv CaMgO_full_tmp_3231.f9 CaMgO_full_3231.f9
mv CaMgO_full_tmp_3231.gui CaMgO_full_3231.gui
mv CaMgO_full_tmp_3232.f9 CaMgO_full_3232.f9
mv CaMgO_full_tmp_3232.gui CaMgO_full_3232.gui
mv CaMgO_full_tmp_3233.f9 CaMgO_full_3233.f9
mv CaMgO_full_tmp_3233.gui CaMgO_full_3233.gui
mv CaMgO_full_tmp_3234.f9 CaMgO_full_3234.f9
mv CaMgO_full_tmp_3234.gui CaMgO_full_3234.gui
mv CaMgO_full_tmp_3235.f9 CaMgO_full_3235.f9
mv CaMgO_full_tmp_3235.gui CaMgO_full_3235.gui
mv CaMgO_full_tmp_3236.f9 CaMgO_full_3236.f9
mv CaMgO_full_tmp_3236.gui CaMgO_full_3236.gui
mv CaMgO_full_tmp_3237.f9 CaMgO_full_3237.f9
mv CaMgO_full_tmp_3237.gui CaMgO_full_3237.gui
mv CaMgO_full_tmp_3238.f9 CaMgO_full_3238.f9
mv CaMgO_full_tmp_3238.gui CaMgO_full_3238.gui
mv CaMgO_full_tmp_3239.f9 CaMgO_full_3239.f9
mv CaMgO_full_tmp_3239.gui CaMgO_full_3239.gui
mv CaMgO_full_tmp_3240.f9 CaMgO_full_3240.f9
mv CaMgO_full_tmp_3240.gui CaMgO_full_3240.gui
mv CaMgO_full_tmp_3241.f9 CaMgO_full_3241.f9
mv CaMgO_full_tmp_3241.gui CaMgO_full_3241.gui
mv CaMgO_full_tmp_3242.f9 CaMgO_full_3242.f9
mv CaMgO_full_tmp_3242.gui CaMgO_full_3242.gui
mv CaMgO_full_tmp_3243.f9 CaMgO_full_3243.f9
mv CaMgO_full_tmp_3243.gui CaMgO_full_3243.gui
mv CaMgO_full_tmp_3244.f9 CaMgO_full_3244.f9
mv CaMgO_full_tmp_3244.gui CaMgO_full_3244.gui
mv CaMgO_full_tmp_3245.f9 CaMgO_full_3245.f9
mv CaMgO_full_tmp_3245.gui CaMgO_full_3245.gui
mv CaMgO_full_tmp_3246.f9 CaMgO_full_3246.f9
mv CaMgO_full_tmp_3246.gui CaMgO_full_3246.gui
mv CaMgO_full_tmp_3247.f9 CaMgO_full_3247.f9
mv CaMgO_full_tmp_3247.gui CaMgO_full_3247.gui
mv CaMgO_full_tmp_3248.f9 CaMgO_full_3248.f9
mv CaMgO_full_tmp_3248.gui CaMgO_full_3248.gui
mv CaMgO_full_tmp_3249.f9 CaMgO_full_3249.f9
mv CaMgO_full_tmp_3249.gui CaMgO_full_3249.gui
mv CaMgO_full_tmp_3250.f9 CaMgO_full_3250.f9
mv CaMgO_full_tmp_3250.gui CaMgO_full_3250.gui
mv CaMgO_full_tmp_3251.f9 CaMgO_full_3251.f9
mv CaMgO_full_tmp_3251.gui CaMgO_full_3251.gui
mv CaMgO_full_tmp_3252.f9 CaMgO_full_3252.f9
mv CaMgO_full_tmp_3252.gui CaMgO_full_3252.gui
mv CaMgO_full_tmp_3253.f9 CaMgO_full_3253.f9
mv CaMgO_full_tmp_3253.gui CaMgO_full_3253.gui
mv CaMgO_full_tmp_3254.f9 CaMgO_full_3254.f9
mv CaMgO_full_tmp_3254.gui CaMgO_full_3254.gui
mv CaMgO_full_tmp_3255.f9 CaMgO_full_3255.f9
mv CaMgO_full_tmp_3255.gui CaMgO_full_3255.gui
mv CaMgO_full_tmp_3256.f9 CaMgO_full_3256.f9
mv CaMgO_full_tmp_3256.gui CaMgO_full_3256.gui
mv CaMgO_full_tmp_3257.f9 CaMgO_full_3257.f9
mv CaMgO_full_tmp_3257.gui CaMgO_full_3257.gui
mv CaMgO_full_tmp_3258.f9 CaMgO_full_3258.f9
mv CaMgO_full_tmp_3258.gui CaMgO_full_3258.gui
mv CaMgO_full_tmp_3259.f9 CaMgO_full_3259.f9
mv CaMgO_full_tmp_3259.gui CaMgO_full_3259.gui
mv CaMgO_full_tmp_3260.f9 CaMgO_full_3260.f9
mv CaMgO_full_tmp_3260.gui CaMgO_full_3260.gui
mv CaMgO_full_tmp_3261.f9 CaMgO_full_3261.f9
mv CaMgO_full_tmp_3261.gui CaMgO_full_3261.gui
mv CaMgO_full_tmp_3262.f9 CaMgO_full_3262.f9
mv CaMgO_full_tmp_3262.gui CaMgO_full_3262.gui
mv CaMgO_full_tmp_3263.f9 CaMgO_full_3263.f9
mv CaMgO_full_tmp_3263.gui CaMgO_full_3263.gui
mv CaMgO_full_tmp_3264.f9 CaMgO_full_3264.f9
mv CaMgO_full_tmp_3264.gui CaMgO_full_3264.gui
mv CaMgO_full_tmp_3265.f9 CaMgO_full_3265.f9
mv CaMgO_full_tmp_3265.gui CaMgO_full_3265.gui
mv CaMgO_full_tmp_3266.f9 CaMgO_full_3266.f9
mv CaMgO_full_tmp_3266.gui CaMgO_full_3266.gui
mv CaMgO_full_tmp_3267.f9 CaMgO_full_3267.f9
mv CaMgO_full_tmp_3267.gui CaMgO_full_3267.gui
mv CaMgO_full_tmp_3268.f9 CaMgO_full_3268.f9
mv CaMgO_full_tmp_3268.gui CaMgO_full_3268.gui
mv CaMgO_full_tmp_3269.f9 CaMgO_full_3269.f9
mv CaMgO_full_tmp_3269.gui CaMgO_full_3269.gui
mv CaMgO_full_tmp_3270.f9 CaMgO_full_3270.f9
mv CaMgO_full_tmp_3270.gui CaMgO_full_3270.gui
mv CaMgO_full_tmp_3271.f9 CaMgO_full_3271.f9
mv CaMgO_full_tmp_3271.gui CaMgO_full_3271.gui
mv CaMgO_full_tmp_3272.f9 CaMgO_full_3272.f9
mv CaMgO_full_tmp_3272.gui CaMgO_full_3272.gui
mv CaMgO_full_tmp_3273.f9 CaMgO_full_3273.f9
mv CaMgO_full_tmp_3273.gui CaMgO_full_3273.gui
mv CaMgO_full_tmp_3274.f9 CaMgO_full_3274.f9
mv CaMgO_full_tmp_3274.gui CaMgO_full_3274.gui
mv CaMgO_full_tmp_3275.f9 CaMgO_full_3275.f9
mv CaMgO_full_tmp_3275.gui CaMgO_full_3275.gui
mv CaMgO_full_tmp_3276.f9 CaMgO_full_3276.f9
mv CaMgO_full_tmp_3276.gui CaMgO_full_3276.gui
mv CaMgO_full_tmp_3277.f9 CaMgO_full_3277.f9
mv CaMgO_full_tmp_3277.gui CaMgO_full_3277.gui
mv CaMgO_full_tmp_3278.f9 CaMgO_full_3278.f9
mv CaMgO_full_tmp_3278.gui CaMgO_full_3278.gui
mv CaMgO_full_tmp_3279.f9 CaMgO_full_3279.f9
mv CaMgO_full_tmp_3279.gui CaMgO_full_3279.gui
mv CaMgO_full_tmp_3280.f9 CaMgO_full_3280.f9
mv CaMgO_full_tmp_3280.gui CaMgO_full_3280.gui
mv CaMgO_full_tmp_3281.f9 CaMgO_full_3281.f9
mv CaMgO_full_tmp_3281.gui CaMgO_full_3281.gui
mv CaMgO_full_tmp_3282.f9 CaMgO_full_3282.f9
mv CaMgO_full_tmp_3282.gui CaMgO_full_3282.gui
mv CaMgO_full_tmp_3283.f9 CaMgO_full_3283.f9
mv CaMgO_full_tmp_3283.gui CaMgO_full_3283.gui
mv CaMgO_full_tmp_3284.f9 CaMgO_full_3284.f9
mv CaMgO_full_tmp_3284.gui CaMgO_full_3284.gui
mv CaMgO_full_tmp_3285.f9 CaMgO_full_3285.f9
mv CaMgO_full_tmp_3285.gui CaMgO_full_3285.gui
mv CaMgO_full_tmp_3286.f9 CaMgO_full_3286.f9
mv CaMgO_full_tmp_3286.gui CaMgO_full_3286.gui
mv CaMgO_full_tmp_3287.f9 CaMgO_full_3287.f9
mv CaMgO_full_tmp_3287.gui CaMgO_full_3287.gui
mv CaMgO_full_tmp_3288.f9 CaMgO_full_3288.f9
mv CaMgO_full_tmp_3288.gui CaMgO_full_3288.gui
mv CaMgO_full_tmp_3289.f9 CaMgO_full_3289.f9
mv CaMgO_full_tmp_3289.gui CaMgO_full_3289.gui
mv CaMgO_full_tmp_3290.f9 CaMgO_full_3290.f9
mv CaMgO_full_tmp_3290.gui CaMgO_full_3290.gui
mv CaMgO_full_tmp_3291.f9 CaMgO_full_3291.f9
mv CaMgO_full_tmp_3291.gui CaMgO_full_3291.gui
mv CaMgO_full_tmp_3292.f9 CaMgO_full_3292.f9
mv CaMgO_full_tmp_3292.gui CaMgO_full_3292.gui
mv CaMgO_full_tmp_3293.f9 CaMgO_full_3293.f9
mv CaMgO_full_tmp_3293.gui CaMgO_full_3293.gui
mv CaMgO_full_tmp_3294.f9 CaMgO_full_3294.f9
mv CaMgO_full_tmp_3294.gui CaMgO_full_3294.gui
mv CaMgO_full_tmp_3295.f9 CaMgO_full_3295.f9
mv CaMgO_full_tmp_3295.gui CaMgO_full_3295.gui
mv CaMgO_full_tmp_3296.f9 CaMgO_full_3296.f9
mv CaMgO_full_tmp_3296.gui CaMgO_full_3296.gui
mv CaMgO_full_tmp_3297.f9 CaMgO_full_3297.f9
mv CaMgO_full_tmp_3297.gui CaMgO_full_3297.gui
mv CaMgO_full_tmp_3298.f9 CaMgO_full_3298.f9
mv CaMgO_full_tmp_3298.gui CaMgO_full_3298.gui
mv CaMgO_full_tmp_3299.f9 CaMgO_full_3299.f9
mv CaMgO_full_tmp_3299.gui CaMgO_full_3299.gui
