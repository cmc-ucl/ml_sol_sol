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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3400 CaMgO_full_tmp_3400 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3401 CaMgO_full_tmp_3401 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3402 CaMgO_full_tmp_3402 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3403 CaMgO_full_tmp_3403 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3404 CaMgO_full_tmp_3404 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3405 CaMgO_full_tmp_3405 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3406 CaMgO_full_tmp_3406 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3407 CaMgO_full_tmp_3407 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3408 CaMgO_full_tmp_3408 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3409 CaMgO_full_tmp_3409 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3410 CaMgO_full_tmp_3410 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3411 CaMgO_full_tmp_3411 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3412 CaMgO_full_tmp_3412 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3413 CaMgO_full_tmp_3413 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3414 CaMgO_full_tmp_3414 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3415 CaMgO_full_tmp_3415 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3416 CaMgO_full_tmp_3416 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3417 CaMgO_full_tmp_3417 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3418 CaMgO_full_tmp_3418 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3419 CaMgO_full_tmp_3419 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3420 CaMgO_full_tmp_3420 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3421 CaMgO_full_tmp_3421 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3422 CaMgO_full_tmp_3422 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3423 CaMgO_full_tmp_3423 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3424 CaMgO_full_tmp_3424 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3425 CaMgO_full_tmp_3425 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3426 CaMgO_full_tmp_3426 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3427 CaMgO_full_tmp_3427 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3428 CaMgO_full_tmp_3428 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3429 CaMgO_full_tmp_3429 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3430 CaMgO_full_tmp_3430 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3431 CaMgO_full_tmp_3431 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3432 CaMgO_full_tmp_3432 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3433 CaMgO_full_tmp_3433 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3434 CaMgO_full_tmp_3434 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3435 CaMgO_full_tmp_3435 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3436 CaMgO_full_tmp_3436 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3437 CaMgO_full_tmp_3437 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3438 CaMgO_full_tmp_3438 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3439 CaMgO_full_tmp_3439 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3440 CaMgO_full_tmp_3440 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3441 CaMgO_full_tmp_3441 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3442 CaMgO_full_tmp_3442 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3443 CaMgO_full_tmp_3443 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3444 CaMgO_full_tmp_3444 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3445 CaMgO_full_tmp_3445 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3446 CaMgO_full_tmp_3446 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3447 CaMgO_full_tmp_3447 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3448 CaMgO_full_tmp_3448 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3449 CaMgO_full_tmp_3449 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3450 CaMgO_full_tmp_3450 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3451 CaMgO_full_tmp_3451 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3452 CaMgO_full_tmp_3452 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3453 CaMgO_full_tmp_3453 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3454 CaMgO_full_tmp_3454 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3455 CaMgO_full_tmp_3455 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3456 CaMgO_full_tmp_3456 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3457 CaMgO_full_tmp_3457 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3458 CaMgO_full_tmp_3458 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3459 CaMgO_full_tmp_3459 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3460 CaMgO_full_tmp_3460 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3461 CaMgO_full_tmp_3461 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3462 CaMgO_full_tmp_3462 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3463 CaMgO_full_tmp_3463 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3464 CaMgO_full_tmp_3464 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3465 CaMgO_full_tmp_3465 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3466 CaMgO_full_tmp_3466 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3467 CaMgO_full_tmp_3467 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3468 CaMgO_full_tmp_3468 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3469 CaMgO_full_tmp_3469 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3470 CaMgO_full_tmp_3470 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3471 CaMgO_full_tmp_3471 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3472 CaMgO_full_tmp_3472 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3473 CaMgO_full_tmp_3473 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3474 CaMgO_full_tmp_3474 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3475 CaMgO_full_tmp_3475 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3476 CaMgO_full_tmp_3476 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3477 CaMgO_full_tmp_3477 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3478 CaMgO_full_tmp_3478 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3479 CaMgO_full_tmp_3479 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3480 CaMgO_full_tmp_3480 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3481 CaMgO_full_tmp_3481 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3482 CaMgO_full_tmp_3482 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3483 CaMgO_full_tmp_3483 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3484 CaMgO_full_tmp_3484 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3485 CaMgO_full_tmp_3485 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3486 CaMgO_full_tmp_3486 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3487 CaMgO_full_tmp_3487 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3488 CaMgO_full_tmp_3488 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3489 CaMgO_full_tmp_3489 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3490 CaMgO_full_tmp_3490 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3491 CaMgO_full_tmp_3491 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3492 CaMgO_full_tmp_3492 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3493 CaMgO_full_tmp_3493 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3494 CaMgO_full_tmp_3494 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3495 CaMgO_full_tmp_3495 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3496 CaMgO_full_tmp_3496 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3497 CaMgO_full_tmp_3497 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3498 CaMgO_full_tmp_3498 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3499 CaMgO_full_tmp_3499 &

wait
mv CaMgO_full_tmp_3400.f9 CaMgO_full_3400.f9
mv CaMgO_full_tmp_3400.gui CaMgO_full_3400.gui
mv CaMgO_full_tmp_3401.f9 CaMgO_full_3401.f9
mv CaMgO_full_tmp_3401.gui CaMgO_full_3401.gui
mv CaMgO_full_tmp_3402.f9 CaMgO_full_3402.f9
mv CaMgO_full_tmp_3402.gui CaMgO_full_3402.gui
mv CaMgO_full_tmp_3403.f9 CaMgO_full_3403.f9
mv CaMgO_full_tmp_3403.gui CaMgO_full_3403.gui
mv CaMgO_full_tmp_3404.f9 CaMgO_full_3404.f9
mv CaMgO_full_tmp_3404.gui CaMgO_full_3404.gui
mv CaMgO_full_tmp_3405.f9 CaMgO_full_3405.f9
mv CaMgO_full_tmp_3405.gui CaMgO_full_3405.gui
mv CaMgO_full_tmp_3406.f9 CaMgO_full_3406.f9
mv CaMgO_full_tmp_3406.gui CaMgO_full_3406.gui
mv CaMgO_full_tmp_3407.f9 CaMgO_full_3407.f9
mv CaMgO_full_tmp_3407.gui CaMgO_full_3407.gui
mv CaMgO_full_tmp_3408.f9 CaMgO_full_3408.f9
mv CaMgO_full_tmp_3408.gui CaMgO_full_3408.gui
mv CaMgO_full_tmp_3409.f9 CaMgO_full_3409.f9
mv CaMgO_full_tmp_3409.gui CaMgO_full_3409.gui
mv CaMgO_full_tmp_3410.f9 CaMgO_full_3410.f9
mv CaMgO_full_tmp_3410.gui CaMgO_full_3410.gui
mv CaMgO_full_tmp_3411.f9 CaMgO_full_3411.f9
mv CaMgO_full_tmp_3411.gui CaMgO_full_3411.gui
mv CaMgO_full_tmp_3412.f9 CaMgO_full_3412.f9
mv CaMgO_full_tmp_3412.gui CaMgO_full_3412.gui
mv CaMgO_full_tmp_3413.f9 CaMgO_full_3413.f9
mv CaMgO_full_tmp_3413.gui CaMgO_full_3413.gui
mv CaMgO_full_tmp_3414.f9 CaMgO_full_3414.f9
mv CaMgO_full_tmp_3414.gui CaMgO_full_3414.gui
mv CaMgO_full_tmp_3415.f9 CaMgO_full_3415.f9
mv CaMgO_full_tmp_3415.gui CaMgO_full_3415.gui
mv CaMgO_full_tmp_3416.f9 CaMgO_full_3416.f9
mv CaMgO_full_tmp_3416.gui CaMgO_full_3416.gui
mv CaMgO_full_tmp_3417.f9 CaMgO_full_3417.f9
mv CaMgO_full_tmp_3417.gui CaMgO_full_3417.gui
mv CaMgO_full_tmp_3418.f9 CaMgO_full_3418.f9
mv CaMgO_full_tmp_3418.gui CaMgO_full_3418.gui
mv CaMgO_full_tmp_3419.f9 CaMgO_full_3419.f9
mv CaMgO_full_tmp_3419.gui CaMgO_full_3419.gui
mv CaMgO_full_tmp_3420.f9 CaMgO_full_3420.f9
mv CaMgO_full_tmp_3420.gui CaMgO_full_3420.gui
mv CaMgO_full_tmp_3421.f9 CaMgO_full_3421.f9
mv CaMgO_full_tmp_3421.gui CaMgO_full_3421.gui
mv CaMgO_full_tmp_3422.f9 CaMgO_full_3422.f9
mv CaMgO_full_tmp_3422.gui CaMgO_full_3422.gui
mv CaMgO_full_tmp_3423.f9 CaMgO_full_3423.f9
mv CaMgO_full_tmp_3423.gui CaMgO_full_3423.gui
mv CaMgO_full_tmp_3424.f9 CaMgO_full_3424.f9
mv CaMgO_full_tmp_3424.gui CaMgO_full_3424.gui
mv CaMgO_full_tmp_3425.f9 CaMgO_full_3425.f9
mv CaMgO_full_tmp_3425.gui CaMgO_full_3425.gui
mv CaMgO_full_tmp_3426.f9 CaMgO_full_3426.f9
mv CaMgO_full_tmp_3426.gui CaMgO_full_3426.gui
mv CaMgO_full_tmp_3427.f9 CaMgO_full_3427.f9
mv CaMgO_full_tmp_3427.gui CaMgO_full_3427.gui
mv CaMgO_full_tmp_3428.f9 CaMgO_full_3428.f9
mv CaMgO_full_tmp_3428.gui CaMgO_full_3428.gui
mv CaMgO_full_tmp_3429.f9 CaMgO_full_3429.f9
mv CaMgO_full_tmp_3429.gui CaMgO_full_3429.gui
mv CaMgO_full_tmp_3430.f9 CaMgO_full_3430.f9
mv CaMgO_full_tmp_3430.gui CaMgO_full_3430.gui
mv CaMgO_full_tmp_3431.f9 CaMgO_full_3431.f9
mv CaMgO_full_tmp_3431.gui CaMgO_full_3431.gui
mv CaMgO_full_tmp_3432.f9 CaMgO_full_3432.f9
mv CaMgO_full_tmp_3432.gui CaMgO_full_3432.gui
mv CaMgO_full_tmp_3433.f9 CaMgO_full_3433.f9
mv CaMgO_full_tmp_3433.gui CaMgO_full_3433.gui
mv CaMgO_full_tmp_3434.f9 CaMgO_full_3434.f9
mv CaMgO_full_tmp_3434.gui CaMgO_full_3434.gui
mv CaMgO_full_tmp_3435.f9 CaMgO_full_3435.f9
mv CaMgO_full_tmp_3435.gui CaMgO_full_3435.gui
mv CaMgO_full_tmp_3436.f9 CaMgO_full_3436.f9
mv CaMgO_full_tmp_3436.gui CaMgO_full_3436.gui
mv CaMgO_full_tmp_3437.f9 CaMgO_full_3437.f9
mv CaMgO_full_tmp_3437.gui CaMgO_full_3437.gui
mv CaMgO_full_tmp_3438.f9 CaMgO_full_3438.f9
mv CaMgO_full_tmp_3438.gui CaMgO_full_3438.gui
mv CaMgO_full_tmp_3439.f9 CaMgO_full_3439.f9
mv CaMgO_full_tmp_3439.gui CaMgO_full_3439.gui
mv CaMgO_full_tmp_3440.f9 CaMgO_full_3440.f9
mv CaMgO_full_tmp_3440.gui CaMgO_full_3440.gui
mv CaMgO_full_tmp_3441.f9 CaMgO_full_3441.f9
mv CaMgO_full_tmp_3441.gui CaMgO_full_3441.gui
mv CaMgO_full_tmp_3442.f9 CaMgO_full_3442.f9
mv CaMgO_full_tmp_3442.gui CaMgO_full_3442.gui
mv CaMgO_full_tmp_3443.f9 CaMgO_full_3443.f9
mv CaMgO_full_tmp_3443.gui CaMgO_full_3443.gui
mv CaMgO_full_tmp_3444.f9 CaMgO_full_3444.f9
mv CaMgO_full_tmp_3444.gui CaMgO_full_3444.gui
mv CaMgO_full_tmp_3445.f9 CaMgO_full_3445.f9
mv CaMgO_full_tmp_3445.gui CaMgO_full_3445.gui
mv CaMgO_full_tmp_3446.f9 CaMgO_full_3446.f9
mv CaMgO_full_tmp_3446.gui CaMgO_full_3446.gui
mv CaMgO_full_tmp_3447.f9 CaMgO_full_3447.f9
mv CaMgO_full_tmp_3447.gui CaMgO_full_3447.gui
mv CaMgO_full_tmp_3448.f9 CaMgO_full_3448.f9
mv CaMgO_full_tmp_3448.gui CaMgO_full_3448.gui
mv CaMgO_full_tmp_3449.f9 CaMgO_full_3449.f9
mv CaMgO_full_tmp_3449.gui CaMgO_full_3449.gui
mv CaMgO_full_tmp_3450.f9 CaMgO_full_3450.f9
mv CaMgO_full_tmp_3450.gui CaMgO_full_3450.gui
mv CaMgO_full_tmp_3451.f9 CaMgO_full_3451.f9
mv CaMgO_full_tmp_3451.gui CaMgO_full_3451.gui
mv CaMgO_full_tmp_3452.f9 CaMgO_full_3452.f9
mv CaMgO_full_tmp_3452.gui CaMgO_full_3452.gui
mv CaMgO_full_tmp_3453.f9 CaMgO_full_3453.f9
mv CaMgO_full_tmp_3453.gui CaMgO_full_3453.gui
mv CaMgO_full_tmp_3454.f9 CaMgO_full_3454.f9
mv CaMgO_full_tmp_3454.gui CaMgO_full_3454.gui
mv CaMgO_full_tmp_3455.f9 CaMgO_full_3455.f9
mv CaMgO_full_tmp_3455.gui CaMgO_full_3455.gui
mv CaMgO_full_tmp_3456.f9 CaMgO_full_3456.f9
mv CaMgO_full_tmp_3456.gui CaMgO_full_3456.gui
mv CaMgO_full_tmp_3457.f9 CaMgO_full_3457.f9
mv CaMgO_full_tmp_3457.gui CaMgO_full_3457.gui
mv CaMgO_full_tmp_3458.f9 CaMgO_full_3458.f9
mv CaMgO_full_tmp_3458.gui CaMgO_full_3458.gui
mv CaMgO_full_tmp_3459.f9 CaMgO_full_3459.f9
mv CaMgO_full_tmp_3459.gui CaMgO_full_3459.gui
mv CaMgO_full_tmp_3460.f9 CaMgO_full_3460.f9
mv CaMgO_full_tmp_3460.gui CaMgO_full_3460.gui
mv CaMgO_full_tmp_3461.f9 CaMgO_full_3461.f9
mv CaMgO_full_tmp_3461.gui CaMgO_full_3461.gui
mv CaMgO_full_tmp_3462.f9 CaMgO_full_3462.f9
mv CaMgO_full_tmp_3462.gui CaMgO_full_3462.gui
mv CaMgO_full_tmp_3463.f9 CaMgO_full_3463.f9
mv CaMgO_full_tmp_3463.gui CaMgO_full_3463.gui
mv CaMgO_full_tmp_3464.f9 CaMgO_full_3464.f9
mv CaMgO_full_tmp_3464.gui CaMgO_full_3464.gui
mv CaMgO_full_tmp_3465.f9 CaMgO_full_3465.f9
mv CaMgO_full_tmp_3465.gui CaMgO_full_3465.gui
mv CaMgO_full_tmp_3466.f9 CaMgO_full_3466.f9
mv CaMgO_full_tmp_3466.gui CaMgO_full_3466.gui
mv CaMgO_full_tmp_3467.f9 CaMgO_full_3467.f9
mv CaMgO_full_tmp_3467.gui CaMgO_full_3467.gui
mv CaMgO_full_tmp_3468.f9 CaMgO_full_3468.f9
mv CaMgO_full_tmp_3468.gui CaMgO_full_3468.gui
mv CaMgO_full_tmp_3469.f9 CaMgO_full_3469.f9
mv CaMgO_full_tmp_3469.gui CaMgO_full_3469.gui
mv CaMgO_full_tmp_3470.f9 CaMgO_full_3470.f9
mv CaMgO_full_tmp_3470.gui CaMgO_full_3470.gui
mv CaMgO_full_tmp_3471.f9 CaMgO_full_3471.f9
mv CaMgO_full_tmp_3471.gui CaMgO_full_3471.gui
mv CaMgO_full_tmp_3472.f9 CaMgO_full_3472.f9
mv CaMgO_full_tmp_3472.gui CaMgO_full_3472.gui
mv CaMgO_full_tmp_3473.f9 CaMgO_full_3473.f9
mv CaMgO_full_tmp_3473.gui CaMgO_full_3473.gui
mv CaMgO_full_tmp_3474.f9 CaMgO_full_3474.f9
mv CaMgO_full_tmp_3474.gui CaMgO_full_3474.gui
mv CaMgO_full_tmp_3475.f9 CaMgO_full_3475.f9
mv CaMgO_full_tmp_3475.gui CaMgO_full_3475.gui
mv CaMgO_full_tmp_3476.f9 CaMgO_full_3476.f9
mv CaMgO_full_tmp_3476.gui CaMgO_full_3476.gui
mv CaMgO_full_tmp_3477.f9 CaMgO_full_3477.f9
mv CaMgO_full_tmp_3477.gui CaMgO_full_3477.gui
mv CaMgO_full_tmp_3478.f9 CaMgO_full_3478.f9
mv CaMgO_full_tmp_3478.gui CaMgO_full_3478.gui
mv CaMgO_full_tmp_3479.f9 CaMgO_full_3479.f9
mv CaMgO_full_tmp_3479.gui CaMgO_full_3479.gui
mv CaMgO_full_tmp_3480.f9 CaMgO_full_3480.f9
mv CaMgO_full_tmp_3480.gui CaMgO_full_3480.gui
mv CaMgO_full_tmp_3481.f9 CaMgO_full_3481.f9
mv CaMgO_full_tmp_3481.gui CaMgO_full_3481.gui
mv CaMgO_full_tmp_3482.f9 CaMgO_full_3482.f9
mv CaMgO_full_tmp_3482.gui CaMgO_full_3482.gui
mv CaMgO_full_tmp_3483.f9 CaMgO_full_3483.f9
mv CaMgO_full_tmp_3483.gui CaMgO_full_3483.gui
mv CaMgO_full_tmp_3484.f9 CaMgO_full_3484.f9
mv CaMgO_full_tmp_3484.gui CaMgO_full_3484.gui
mv CaMgO_full_tmp_3485.f9 CaMgO_full_3485.f9
mv CaMgO_full_tmp_3485.gui CaMgO_full_3485.gui
mv CaMgO_full_tmp_3486.f9 CaMgO_full_3486.f9
mv CaMgO_full_tmp_3486.gui CaMgO_full_3486.gui
mv CaMgO_full_tmp_3487.f9 CaMgO_full_3487.f9
mv CaMgO_full_tmp_3487.gui CaMgO_full_3487.gui
mv CaMgO_full_tmp_3488.f9 CaMgO_full_3488.f9
mv CaMgO_full_tmp_3488.gui CaMgO_full_3488.gui
mv CaMgO_full_tmp_3489.f9 CaMgO_full_3489.f9
mv CaMgO_full_tmp_3489.gui CaMgO_full_3489.gui
mv CaMgO_full_tmp_3490.f9 CaMgO_full_3490.f9
mv CaMgO_full_tmp_3490.gui CaMgO_full_3490.gui
mv CaMgO_full_tmp_3491.f9 CaMgO_full_3491.f9
mv CaMgO_full_tmp_3491.gui CaMgO_full_3491.gui
mv CaMgO_full_tmp_3492.f9 CaMgO_full_3492.f9
mv CaMgO_full_tmp_3492.gui CaMgO_full_3492.gui
mv CaMgO_full_tmp_3493.f9 CaMgO_full_3493.f9
mv CaMgO_full_tmp_3493.gui CaMgO_full_3493.gui
mv CaMgO_full_tmp_3494.f9 CaMgO_full_3494.f9
mv CaMgO_full_tmp_3494.gui CaMgO_full_3494.gui
mv CaMgO_full_tmp_3495.f9 CaMgO_full_3495.f9
mv CaMgO_full_tmp_3495.gui CaMgO_full_3495.gui
mv CaMgO_full_tmp_3496.f9 CaMgO_full_3496.f9
mv CaMgO_full_tmp_3496.gui CaMgO_full_3496.gui
mv CaMgO_full_tmp_3497.f9 CaMgO_full_3497.f9
mv CaMgO_full_tmp_3497.gui CaMgO_full_3497.gui
mv CaMgO_full_tmp_3498.f9 CaMgO_full_3498.f9
mv CaMgO_full_tmp_3498.gui CaMgO_full_3498.gui
mv CaMgO_full_tmp_3499.f9 CaMgO_full_3499.f9
mv CaMgO_full_tmp_3499.gui CaMgO_full_3499.gui
