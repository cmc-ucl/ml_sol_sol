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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3300 CaMgO_full_tmp_3300 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3301 CaMgO_full_tmp_3301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3302 CaMgO_full_tmp_3302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3303 CaMgO_full_tmp_3303 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3304 CaMgO_full_tmp_3304 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3305 CaMgO_full_tmp_3305 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3306 CaMgO_full_tmp_3306 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3307 CaMgO_full_tmp_3307 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3308 CaMgO_full_tmp_3308 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3309 CaMgO_full_tmp_3309 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3310 CaMgO_full_tmp_3310 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3311 CaMgO_full_tmp_3311 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3312 CaMgO_full_tmp_3312 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3313 CaMgO_full_tmp_3313 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3314 CaMgO_full_tmp_3314 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3315 CaMgO_full_tmp_3315 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3316 CaMgO_full_tmp_3316 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3317 CaMgO_full_tmp_3317 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3318 CaMgO_full_tmp_3318 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3319 CaMgO_full_tmp_3319 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3320 CaMgO_full_tmp_3320 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3321 CaMgO_full_tmp_3321 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3322 CaMgO_full_tmp_3322 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3323 CaMgO_full_tmp_3323 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3324 CaMgO_full_tmp_3324 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3325 CaMgO_full_tmp_3325 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3326 CaMgO_full_tmp_3326 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3327 CaMgO_full_tmp_3327 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3328 CaMgO_full_tmp_3328 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3329 CaMgO_full_tmp_3329 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3330 CaMgO_full_tmp_3330 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3331 CaMgO_full_tmp_3331 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3332 CaMgO_full_tmp_3332 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3333 CaMgO_full_tmp_3333 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3334 CaMgO_full_tmp_3334 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3335 CaMgO_full_tmp_3335 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3336 CaMgO_full_tmp_3336 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3337 CaMgO_full_tmp_3337 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3338 CaMgO_full_tmp_3338 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3339 CaMgO_full_tmp_3339 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3340 CaMgO_full_tmp_3340 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3341 CaMgO_full_tmp_3341 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3342 CaMgO_full_tmp_3342 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3343 CaMgO_full_tmp_3343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3344 CaMgO_full_tmp_3344 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3345 CaMgO_full_tmp_3345 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3346 CaMgO_full_tmp_3346 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3347 CaMgO_full_tmp_3347 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3348 CaMgO_full_tmp_3348 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3349 CaMgO_full_tmp_3349 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3350 CaMgO_full_tmp_3350 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3351 CaMgO_full_tmp_3351 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3352 CaMgO_full_tmp_3352 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3353 CaMgO_full_tmp_3353 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3354 CaMgO_full_tmp_3354 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3355 CaMgO_full_tmp_3355 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3356 CaMgO_full_tmp_3356 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3357 CaMgO_full_tmp_3357 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3358 CaMgO_full_tmp_3358 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3359 CaMgO_full_tmp_3359 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3360 CaMgO_full_tmp_3360 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3361 CaMgO_full_tmp_3361 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3362 CaMgO_full_tmp_3362 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3363 CaMgO_full_tmp_3363 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3364 CaMgO_full_tmp_3364 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3365 CaMgO_full_tmp_3365 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3366 CaMgO_full_tmp_3366 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3367 CaMgO_full_tmp_3367 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3368 CaMgO_full_tmp_3368 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3369 CaMgO_full_tmp_3369 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3370 CaMgO_full_tmp_3370 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3371 CaMgO_full_tmp_3371 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3372 CaMgO_full_tmp_3372 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3373 CaMgO_full_tmp_3373 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3374 CaMgO_full_tmp_3374 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3375 CaMgO_full_tmp_3375 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3376 CaMgO_full_tmp_3376 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3377 CaMgO_full_tmp_3377 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3378 CaMgO_full_tmp_3378 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3379 CaMgO_full_tmp_3379 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3380 CaMgO_full_tmp_3380 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3381 CaMgO_full_tmp_3381 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3382 CaMgO_full_tmp_3382 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3383 CaMgO_full_tmp_3383 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3384 CaMgO_full_tmp_3384 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3385 CaMgO_full_tmp_3385 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3386 CaMgO_full_tmp_3386 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3387 CaMgO_full_tmp_3387 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3388 CaMgO_full_tmp_3388 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3389 CaMgO_full_tmp_3389 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3390 CaMgO_full_tmp_3390 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3391 CaMgO_full_tmp_3391 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3392 CaMgO_full_tmp_3392 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3393 CaMgO_full_tmp_3393 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3394 CaMgO_full_tmp_3394 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3395 CaMgO_full_tmp_3395 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3396 CaMgO_full_tmp_3396 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3397 CaMgO_full_tmp_3397 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3398 CaMgO_full_tmp_3398 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3399 CaMgO_full_tmp_3399 &

wait
mv CaMgO_full_tmp_3300.f9 CaMgO_full_3300.f9
mv CaMgO_full_tmp_3300.gui CaMgO_full_3300.gui
mv CaMgO_full_tmp_3301.f9 CaMgO_full_3301.f9
mv CaMgO_full_tmp_3301.gui CaMgO_full_3301.gui
mv CaMgO_full_tmp_3302.f9 CaMgO_full_3302.f9
mv CaMgO_full_tmp_3302.gui CaMgO_full_3302.gui
mv CaMgO_full_tmp_3303.f9 CaMgO_full_3303.f9
mv CaMgO_full_tmp_3303.gui CaMgO_full_3303.gui
mv CaMgO_full_tmp_3304.f9 CaMgO_full_3304.f9
mv CaMgO_full_tmp_3304.gui CaMgO_full_3304.gui
mv CaMgO_full_tmp_3305.f9 CaMgO_full_3305.f9
mv CaMgO_full_tmp_3305.gui CaMgO_full_3305.gui
mv CaMgO_full_tmp_3306.f9 CaMgO_full_3306.f9
mv CaMgO_full_tmp_3306.gui CaMgO_full_3306.gui
mv CaMgO_full_tmp_3307.f9 CaMgO_full_3307.f9
mv CaMgO_full_tmp_3307.gui CaMgO_full_3307.gui
mv CaMgO_full_tmp_3308.f9 CaMgO_full_3308.f9
mv CaMgO_full_tmp_3308.gui CaMgO_full_3308.gui
mv CaMgO_full_tmp_3309.f9 CaMgO_full_3309.f9
mv CaMgO_full_tmp_3309.gui CaMgO_full_3309.gui
mv CaMgO_full_tmp_3310.f9 CaMgO_full_3310.f9
mv CaMgO_full_tmp_3310.gui CaMgO_full_3310.gui
mv CaMgO_full_tmp_3311.f9 CaMgO_full_3311.f9
mv CaMgO_full_tmp_3311.gui CaMgO_full_3311.gui
mv CaMgO_full_tmp_3312.f9 CaMgO_full_3312.f9
mv CaMgO_full_tmp_3312.gui CaMgO_full_3312.gui
mv CaMgO_full_tmp_3313.f9 CaMgO_full_3313.f9
mv CaMgO_full_tmp_3313.gui CaMgO_full_3313.gui
mv CaMgO_full_tmp_3314.f9 CaMgO_full_3314.f9
mv CaMgO_full_tmp_3314.gui CaMgO_full_3314.gui
mv CaMgO_full_tmp_3315.f9 CaMgO_full_3315.f9
mv CaMgO_full_tmp_3315.gui CaMgO_full_3315.gui
mv CaMgO_full_tmp_3316.f9 CaMgO_full_3316.f9
mv CaMgO_full_tmp_3316.gui CaMgO_full_3316.gui
mv CaMgO_full_tmp_3317.f9 CaMgO_full_3317.f9
mv CaMgO_full_tmp_3317.gui CaMgO_full_3317.gui
mv CaMgO_full_tmp_3318.f9 CaMgO_full_3318.f9
mv CaMgO_full_tmp_3318.gui CaMgO_full_3318.gui
mv CaMgO_full_tmp_3319.f9 CaMgO_full_3319.f9
mv CaMgO_full_tmp_3319.gui CaMgO_full_3319.gui
mv CaMgO_full_tmp_3320.f9 CaMgO_full_3320.f9
mv CaMgO_full_tmp_3320.gui CaMgO_full_3320.gui
mv CaMgO_full_tmp_3321.f9 CaMgO_full_3321.f9
mv CaMgO_full_tmp_3321.gui CaMgO_full_3321.gui
mv CaMgO_full_tmp_3322.f9 CaMgO_full_3322.f9
mv CaMgO_full_tmp_3322.gui CaMgO_full_3322.gui
mv CaMgO_full_tmp_3323.f9 CaMgO_full_3323.f9
mv CaMgO_full_tmp_3323.gui CaMgO_full_3323.gui
mv CaMgO_full_tmp_3324.f9 CaMgO_full_3324.f9
mv CaMgO_full_tmp_3324.gui CaMgO_full_3324.gui
mv CaMgO_full_tmp_3325.f9 CaMgO_full_3325.f9
mv CaMgO_full_tmp_3325.gui CaMgO_full_3325.gui
mv CaMgO_full_tmp_3326.f9 CaMgO_full_3326.f9
mv CaMgO_full_tmp_3326.gui CaMgO_full_3326.gui
mv CaMgO_full_tmp_3327.f9 CaMgO_full_3327.f9
mv CaMgO_full_tmp_3327.gui CaMgO_full_3327.gui
mv CaMgO_full_tmp_3328.f9 CaMgO_full_3328.f9
mv CaMgO_full_tmp_3328.gui CaMgO_full_3328.gui
mv CaMgO_full_tmp_3329.f9 CaMgO_full_3329.f9
mv CaMgO_full_tmp_3329.gui CaMgO_full_3329.gui
mv CaMgO_full_tmp_3330.f9 CaMgO_full_3330.f9
mv CaMgO_full_tmp_3330.gui CaMgO_full_3330.gui
mv CaMgO_full_tmp_3331.f9 CaMgO_full_3331.f9
mv CaMgO_full_tmp_3331.gui CaMgO_full_3331.gui
mv CaMgO_full_tmp_3332.f9 CaMgO_full_3332.f9
mv CaMgO_full_tmp_3332.gui CaMgO_full_3332.gui
mv CaMgO_full_tmp_3333.f9 CaMgO_full_3333.f9
mv CaMgO_full_tmp_3333.gui CaMgO_full_3333.gui
mv CaMgO_full_tmp_3334.f9 CaMgO_full_3334.f9
mv CaMgO_full_tmp_3334.gui CaMgO_full_3334.gui
mv CaMgO_full_tmp_3335.f9 CaMgO_full_3335.f9
mv CaMgO_full_tmp_3335.gui CaMgO_full_3335.gui
mv CaMgO_full_tmp_3336.f9 CaMgO_full_3336.f9
mv CaMgO_full_tmp_3336.gui CaMgO_full_3336.gui
mv CaMgO_full_tmp_3337.f9 CaMgO_full_3337.f9
mv CaMgO_full_tmp_3337.gui CaMgO_full_3337.gui
mv CaMgO_full_tmp_3338.f9 CaMgO_full_3338.f9
mv CaMgO_full_tmp_3338.gui CaMgO_full_3338.gui
mv CaMgO_full_tmp_3339.f9 CaMgO_full_3339.f9
mv CaMgO_full_tmp_3339.gui CaMgO_full_3339.gui
mv CaMgO_full_tmp_3340.f9 CaMgO_full_3340.f9
mv CaMgO_full_tmp_3340.gui CaMgO_full_3340.gui
mv CaMgO_full_tmp_3341.f9 CaMgO_full_3341.f9
mv CaMgO_full_tmp_3341.gui CaMgO_full_3341.gui
mv CaMgO_full_tmp_3342.f9 CaMgO_full_3342.f9
mv CaMgO_full_tmp_3342.gui CaMgO_full_3342.gui
mv CaMgO_full_tmp_3343.f9 CaMgO_full_3343.f9
mv CaMgO_full_tmp_3343.gui CaMgO_full_3343.gui
mv CaMgO_full_tmp_3344.f9 CaMgO_full_3344.f9
mv CaMgO_full_tmp_3344.gui CaMgO_full_3344.gui
mv CaMgO_full_tmp_3345.f9 CaMgO_full_3345.f9
mv CaMgO_full_tmp_3345.gui CaMgO_full_3345.gui
mv CaMgO_full_tmp_3346.f9 CaMgO_full_3346.f9
mv CaMgO_full_tmp_3346.gui CaMgO_full_3346.gui
mv CaMgO_full_tmp_3347.f9 CaMgO_full_3347.f9
mv CaMgO_full_tmp_3347.gui CaMgO_full_3347.gui
mv CaMgO_full_tmp_3348.f9 CaMgO_full_3348.f9
mv CaMgO_full_tmp_3348.gui CaMgO_full_3348.gui
mv CaMgO_full_tmp_3349.f9 CaMgO_full_3349.f9
mv CaMgO_full_tmp_3349.gui CaMgO_full_3349.gui
mv CaMgO_full_tmp_3350.f9 CaMgO_full_3350.f9
mv CaMgO_full_tmp_3350.gui CaMgO_full_3350.gui
mv CaMgO_full_tmp_3351.f9 CaMgO_full_3351.f9
mv CaMgO_full_tmp_3351.gui CaMgO_full_3351.gui
mv CaMgO_full_tmp_3352.f9 CaMgO_full_3352.f9
mv CaMgO_full_tmp_3352.gui CaMgO_full_3352.gui
mv CaMgO_full_tmp_3353.f9 CaMgO_full_3353.f9
mv CaMgO_full_tmp_3353.gui CaMgO_full_3353.gui
mv CaMgO_full_tmp_3354.f9 CaMgO_full_3354.f9
mv CaMgO_full_tmp_3354.gui CaMgO_full_3354.gui
mv CaMgO_full_tmp_3355.f9 CaMgO_full_3355.f9
mv CaMgO_full_tmp_3355.gui CaMgO_full_3355.gui
mv CaMgO_full_tmp_3356.f9 CaMgO_full_3356.f9
mv CaMgO_full_tmp_3356.gui CaMgO_full_3356.gui
mv CaMgO_full_tmp_3357.f9 CaMgO_full_3357.f9
mv CaMgO_full_tmp_3357.gui CaMgO_full_3357.gui
mv CaMgO_full_tmp_3358.f9 CaMgO_full_3358.f9
mv CaMgO_full_tmp_3358.gui CaMgO_full_3358.gui
mv CaMgO_full_tmp_3359.f9 CaMgO_full_3359.f9
mv CaMgO_full_tmp_3359.gui CaMgO_full_3359.gui
mv CaMgO_full_tmp_3360.f9 CaMgO_full_3360.f9
mv CaMgO_full_tmp_3360.gui CaMgO_full_3360.gui
mv CaMgO_full_tmp_3361.f9 CaMgO_full_3361.f9
mv CaMgO_full_tmp_3361.gui CaMgO_full_3361.gui
mv CaMgO_full_tmp_3362.f9 CaMgO_full_3362.f9
mv CaMgO_full_tmp_3362.gui CaMgO_full_3362.gui
mv CaMgO_full_tmp_3363.f9 CaMgO_full_3363.f9
mv CaMgO_full_tmp_3363.gui CaMgO_full_3363.gui
mv CaMgO_full_tmp_3364.f9 CaMgO_full_3364.f9
mv CaMgO_full_tmp_3364.gui CaMgO_full_3364.gui
mv CaMgO_full_tmp_3365.f9 CaMgO_full_3365.f9
mv CaMgO_full_tmp_3365.gui CaMgO_full_3365.gui
mv CaMgO_full_tmp_3366.f9 CaMgO_full_3366.f9
mv CaMgO_full_tmp_3366.gui CaMgO_full_3366.gui
mv CaMgO_full_tmp_3367.f9 CaMgO_full_3367.f9
mv CaMgO_full_tmp_3367.gui CaMgO_full_3367.gui
mv CaMgO_full_tmp_3368.f9 CaMgO_full_3368.f9
mv CaMgO_full_tmp_3368.gui CaMgO_full_3368.gui
mv CaMgO_full_tmp_3369.f9 CaMgO_full_3369.f9
mv CaMgO_full_tmp_3369.gui CaMgO_full_3369.gui
mv CaMgO_full_tmp_3370.f9 CaMgO_full_3370.f9
mv CaMgO_full_tmp_3370.gui CaMgO_full_3370.gui
mv CaMgO_full_tmp_3371.f9 CaMgO_full_3371.f9
mv CaMgO_full_tmp_3371.gui CaMgO_full_3371.gui
mv CaMgO_full_tmp_3372.f9 CaMgO_full_3372.f9
mv CaMgO_full_tmp_3372.gui CaMgO_full_3372.gui
mv CaMgO_full_tmp_3373.f9 CaMgO_full_3373.f9
mv CaMgO_full_tmp_3373.gui CaMgO_full_3373.gui
mv CaMgO_full_tmp_3374.f9 CaMgO_full_3374.f9
mv CaMgO_full_tmp_3374.gui CaMgO_full_3374.gui
mv CaMgO_full_tmp_3375.f9 CaMgO_full_3375.f9
mv CaMgO_full_tmp_3375.gui CaMgO_full_3375.gui
mv CaMgO_full_tmp_3376.f9 CaMgO_full_3376.f9
mv CaMgO_full_tmp_3376.gui CaMgO_full_3376.gui
mv CaMgO_full_tmp_3377.f9 CaMgO_full_3377.f9
mv CaMgO_full_tmp_3377.gui CaMgO_full_3377.gui
mv CaMgO_full_tmp_3378.f9 CaMgO_full_3378.f9
mv CaMgO_full_tmp_3378.gui CaMgO_full_3378.gui
mv CaMgO_full_tmp_3379.f9 CaMgO_full_3379.f9
mv CaMgO_full_tmp_3379.gui CaMgO_full_3379.gui
mv CaMgO_full_tmp_3380.f9 CaMgO_full_3380.f9
mv CaMgO_full_tmp_3380.gui CaMgO_full_3380.gui
mv CaMgO_full_tmp_3381.f9 CaMgO_full_3381.f9
mv CaMgO_full_tmp_3381.gui CaMgO_full_3381.gui
mv CaMgO_full_tmp_3382.f9 CaMgO_full_3382.f9
mv CaMgO_full_tmp_3382.gui CaMgO_full_3382.gui
mv CaMgO_full_tmp_3383.f9 CaMgO_full_3383.f9
mv CaMgO_full_tmp_3383.gui CaMgO_full_3383.gui
mv CaMgO_full_tmp_3384.f9 CaMgO_full_3384.f9
mv CaMgO_full_tmp_3384.gui CaMgO_full_3384.gui
mv CaMgO_full_tmp_3385.f9 CaMgO_full_3385.f9
mv CaMgO_full_tmp_3385.gui CaMgO_full_3385.gui
mv CaMgO_full_tmp_3386.f9 CaMgO_full_3386.f9
mv CaMgO_full_tmp_3386.gui CaMgO_full_3386.gui
mv CaMgO_full_tmp_3387.f9 CaMgO_full_3387.f9
mv CaMgO_full_tmp_3387.gui CaMgO_full_3387.gui
mv CaMgO_full_tmp_3388.f9 CaMgO_full_3388.f9
mv CaMgO_full_tmp_3388.gui CaMgO_full_3388.gui
mv CaMgO_full_tmp_3389.f9 CaMgO_full_3389.f9
mv CaMgO_full_tmp_3389.gui CaMgO_full_3389.gui
mv CaMgO_full_tmp_3390.f9 CaMgO_full_3390.f9
mv CaMgO_full_tmp_3390.gui CaMgO_full_3390.gui
mv CaMgO_full_tmp_3391.f9 CaMgO_full_3391.f9
mv CaMgO_full_tmp_3391.gui CaMgO_full_3391.gui
mv CaMgO_full_tmp_3392.f9 CaMgO_full_3392.f9
mv CaMgO_full_tmp_3392.gui CaMgO_full_3392.gui
mv CaMgO_full_tmp_3393.f9 CaMgO_full_3393.f9
mv CaMgO_full_tmp_3393.gui CaMgO_full_3393.gui
mv CaMgO_full_tmp_3394.f9 CaMgO_full_3394.f9
mv CaMgO_full_tmp_3394.gui CaMgO_full_3394.gui
mv CaMgO_full_tmp_3395.f9 CaMgO_full_3395.f9
mv CaMgO_full_tmp_3395.gui CaMgO_full_3395.gui
mv CaMgO_full_tmp_3396.f9 CaMgO_full_3396.f9
mv CaMgO_full_tmp_3396.gui CaMgO_full_3396.gui
mv CaMgO_full_tmp_3397.f9 CaMgO_full_3397.f9
mv CaMgO_full_tmp_3397.gui CaMgO_full_3397.gui
mv CaMgO_full_tmp_3398.f9 CaMgO_full_3398.f9
mv CaMgO_full_tmp_3398.gui CaMgO_full_3398.gui
mv CaMgO_full_tmp_3399.f9 CaMgO_full_3399.f9
mv CaMgO_full_tmp_3399.gui CaMgO_full_3399.gui
