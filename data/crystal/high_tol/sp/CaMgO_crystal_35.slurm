#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-gc-smw    
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3400


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3499


