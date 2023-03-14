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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3301 CaMgO_coord_3301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3302 CaMgO_coord_3302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3303 CaMgO_coord_3303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3304 CaMgO_coord_3304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3305 CaMgO_coord_3305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3306 CaMgO_coord_3306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3307 CaMgO_coord_3307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3308 CaMgO_coord_3308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3309 CaMgO_coord_3309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3310 CaMgO_coord_3310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3311 CaMgO_coord_3311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3312 CaMgO_coord_3312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3313 CaMgO_coord_3313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3314 CaMgO_coord_3314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3315 CaMgO_coord_3315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3316 CaMgO_coord_3316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3317 CaMgO_coord_3317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3318 CaMgO_coord_3318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3319 CaMgO_coord_3319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3320 CaMgO_coord_3320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3321 CaMgO_coord_3321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3322 CaMgO_coord_3322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3323 CaMgO_coord_3323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3324 CaMgO_coord_3324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3325 CaMgO_coord_3325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3326 CaMgO_coord_3326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3327 CaMgO_coord_3327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3328 CaMgO_coord_3328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3329 CaMgO_coord_3329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3330 CaMgO_coord_3330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3331 CaMgO_coord_3331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3332 CaMgO_coord_3332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3333 CaMgO_coord_3333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3334 CaMgO_coord_3334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3335 CaMgO_coord_3335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3336 CaMgO_coord_3336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3337 CaMgO_coord_3337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3338 CaMgO_coord_3338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3339 CaMgO_coord_3339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3340 CaMgO_coord_3340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3341 CaMgO_coord_3341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3342 CaMgO_coord_3342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3343 CaMgO_coord_3343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3344 CaMgO_coord_3344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3345 CaMgO_coord_3345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3346 CaMgO_coord_3346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3347 CaMgO_coord_3347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3348 CaMgO_coord_3348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3349 CaMgO_coord_3349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3350 CaMgO_coord_3350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3351 CaMgO_coord_3351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3352 CaMgO_coord_3352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3353 CaMgO_coord_3353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3354 CaMgO_coord_3354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3355 CaMgO_coord_3355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3356 CaMgO_coord_3356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3357 CaMgO_coord_3357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3358 CaMgO_coord_3358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3359 CaMgO_coord_3359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3360 CaMgO_coord_3360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3361 CaMgO_coord_3361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3362 CaMgO_coord_3362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3363 CaMgO_coord_3363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3364 CaMgO_coord_3364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3365 CaMgO_coord_3365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3366 CaMgO_coord_3366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3367 CaMgO_coord_3367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3368 CaMgO_coord_3368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3369 CaMgO_coord_3369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3370 CaMgO_coord_3370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3371 CaMgO_coord_3371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3372 CaMgO_coord_3372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3373 CaMgO_coord_3373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3374 CaMgO_coord_3374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3375 CaMgO_coord_3375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3376 CaMgO_coord_3376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3377 CaMgO_coord_3377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3378 CaMgO_coord_3378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3379 CaMgO_coord_3379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3380 CaMgO_coord_3380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3381 CaMgO_coord_3381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3382 CaMgO_coord_3382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3383 CaMgO_coord_3383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3384 CaMgO_coord_3384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3385 CaMgO_coord_3385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3386 CaMgO_coord_3386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3387 CaMgO_coord_3387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3388 CaMgO_coord_3388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3389 CaMgO_coord_3389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3390 CaMgO_coord_3390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3391 CaMgO_coord_3391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3392 CaMgO_coord_3392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3393 CaMgO_coord_3393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3394 CaMgO_coord_3394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3395 CaMgO_coord_3395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3396 CaMgO_coord_3396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3397 CaMgO_coord_3397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3398 CaMgO_coord_3398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3399 CaMgO_coord_3399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3399


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3400 CaMgO_coord_3400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3400


