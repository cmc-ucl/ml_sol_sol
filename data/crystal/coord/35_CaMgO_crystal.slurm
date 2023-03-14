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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3401 CaMgO_coord_3401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3402 CaMgO_coord_3402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3403 CaMgO_coord_3403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3404 CaMgO_coord_3404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3405 CaMgO_coord_3405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3406 CaMgO_coord_3406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3407 CaMgO_coord_3407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3408 CaMgO_coord_3408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3409 CaMgO_coord_3409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3410 CaMgO_coord_3410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3411 CaMgO_coord_3411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3412 CaMgO_coord_3412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3413 CaMgO_coord_3413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3414 CaMgO_coord_3414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3415 CaMgO_coord_3415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3416 CaMgO_coord_3416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3417 CaMgO_coord_3417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3418 CaMgO_coord_3418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3419 CaMgO_coord_3419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3420 CaMgO_coord_3420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3421 CaMgO_coord_3421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3422 CaMgO_coord_3422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3423 CaMgO_coord_3423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3424 CaMgO_coord_3424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3425 CaMgO_coord_3425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3426 CaMgO_coord_3426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3427 CaMgO_coord_3427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3428 CaMgO_coord_3428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3429 CaMgO_coord_3429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3430 CaMgO_coord_3430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3431 CaMgO_coord_3431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3432 CaMgO_coord_3432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3433 CaMgO_coord_3433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3434 CaMgO_coord_3434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3435 CaMgO_coord_3435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3436 CaMgO_coord_3436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3437 CaMgO_coord_3437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3438 CaMgO_coord_3438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3439 CaMgO_coord_3439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3440 CaMgO_coord_3440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3441 CaMgO_coord_3441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3442 CaMgO_coord_3442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3443 CaMgO_coord_3443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3444 CaMgO_coord_3444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3445 CaMgO_coord_3445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3446 CaMgO_coord_3446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3447 CaMgO_coord_3447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3448 CaMgO_coord_3448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3449 CaMgO_coord_3449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3450 CaMgO_coord_3450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3451 CaMgO_coord_3451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3452 CaMgO_coord_3452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3453 CaMgO_coord_3453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3454 CaMgO_coord_3454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3455 CaMgO_coord_3455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3456 CaMgO_coord_3456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3457 CaMgO_coord_3457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3458 CaMgO_coord_3458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3459 CaMgO_coord_3459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3460 CaMgO_coord_3460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3461 CaMgO_coord_3461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3462 CaMgO_coord_3462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3463 CaMgO_coord_3463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3464 CaMgO_coord_3464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3465 CaMgO_coord_3465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3466 CaMgO_coord_3466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3467 CaMgO_coord_3467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3468 CaMgO_coord_3468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3469 CaMgO_coord_3469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3470 CaMgO_coord_3470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3471 CaMgO_coord_3471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3472 CaMgO_coord_3472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3473 CaMgO_coord_3473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3474 CaMgO_coord_3474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3475 CaMgO_coord_3475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3476 CaMgO_coord_3476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3477 CaMgO_coord_3477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3478 CaMgO_coord_3478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3479 CaMgO_coord_3479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3480 CaMgO_coord_3480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3481 CaMgO_coord_3481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3482 CaMgO_coord_3482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3483 CaMgO_coord_3483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3484 CaMgO_coord_3484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3485 CaMgO_coord_3485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3486 CaMgO_coord_3486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3487 CaMgO_coord_3487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3488 CaMgO_coord_3488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3489 CaMgO_coord_3489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3490 CaMgO_coord_3490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3491 CaMgO_coord_3491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3492 CaMgO_coord_3492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3493 CaMgO_coord_3493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3494 CaMgO_coord_3494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3495 CaMgO_coord_3495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3496 CaMgO_coord_3496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3497 CaMgO_coord_3497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3498 CaMgO_coord_3498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3499 CaMgO_coord_3499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3499


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3500 CaMgO_coord_3500
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3500


