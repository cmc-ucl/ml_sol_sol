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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3399


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3400


