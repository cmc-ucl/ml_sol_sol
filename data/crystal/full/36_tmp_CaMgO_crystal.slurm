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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3500 CaMgO_full_tmp_3500 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3501 CaMgO_full_tmp_3501 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3502 CaMgO_full_tmp_3502 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3503 CaMgO_full_tmp_3503 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3504 CaMgO_full_tmp_3504 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3505 CaMgO_full_tmp_3505 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3506 CaMgO_full_tmp_3506 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3507 CaMgO_full_tmp_3507 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3508 CaMgO_full_tmp_3508 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3509 CaMgO_full_tmp_3509 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3510 CaMgO_full_tmp_3510 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3511 CaMgO_full_tmp_3511 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3512 CaMgO_full_tmp_3512 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3513 CaMgO_full_tmp_3513 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3514 CaMgO_full_tmp_3514 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3515 CaMgO_full_tmp_3515 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3516 CaMgO_full_tmp_3516 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3517 CaMgO_full_tmp_3517 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3518 CaMgO_full_tmp_3518 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3519 CaMgO_full_tmp_3519 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3520 CaMgO_full_tmp_3520 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3521 CaMgO_full_tmp_3521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3522 CaMgO_full_tmp_3522 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3523 CaMgO_full_tmp_3523 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3524 CaMgO_full_tmp_3524 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3525 CaMgO_full_tmp_3525 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3526 CaMgO_full_tmp_3526 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3527 CaMgO_full_tmp_3527 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3528 CaMgO_full_tmp_3528 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3529 CaMgO_full_tmp_3529 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3530 CaMgO_full_tmp_3530 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3531 CaMgO_full_tmp_3531 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3532 CaMgO_full_tmp_3532 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3533 CaMgO_full_tmp_3533 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3534 CaMgO_full_tmp_3534 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3535 CaMgO_full_tmp_3535 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3536 CaMgO_full_tmp_3536 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3537 CaMgO_full_tmp_3537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3538 CaMgO_full_tmp_3538 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3539 CaMgO_full_tmp_3539 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3540 CaMgO_full_tmp_3540 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3541 CaMgO_full_tmp_3541 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3542 CaMgO_full_tmp_3542 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3543 CaMgO_full_tmp_3543 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3544 CaMgO_full_tmp_3544 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3545 CaMgO_full_tmp_3545 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3546 CaMgO_full_tmp_3546 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3547 CaMgO_full_tmp_3547 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3548 CaMgO_full_tmp_3548 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3549 CaMgO_full_tmp_3549 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3550 CaMgO_full_tmp_3550 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3551 CaMgO_full_tmp_3551 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3552 CaMgO_full_tmp_3552 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3553 CaMgO_full_tmp_3553 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3554 CaMgO_full_tmp_3554 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3555 CaMgO_full_tmp_3555 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3556 CaMgO_full_tmp_3556 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3557 CaMgO_full_tmp_3557 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3558 CaMgO_full_tmp_3558 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3559 CaMgO_full_tmp_3559 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3560 CaMgO_full_tmp_3560 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3561 CaMgO_full_tmp_3561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3562 CaMgO_full_tmp_3562 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3563 CaMgO_full_tmp_3563 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3564 CaMgO_full_tmp_3564 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3565 CaMgO_full_tmp_3565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3566 CaMgO_full_tmp_3566 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3567 CaMgO_full_tmp_3567 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3568 CaMgO_full_tmp_3568 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3569 CaMgO_full_tmp_3569 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3570 CaMgO_full_tmp_3570 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3571 CaMgO_full_tmp_3571 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3572 CaMgO_full_tmp_3572 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3573 CaMgO_full_tmp_3573 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3574 CaMgO_full_tmp_3574 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3575 CaMgO_full_tmp_3575 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3576 CaMgO_full_tmp_3576 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3577 CaMgO_full_tmp_3577 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3578 CaMgO_full_tmp_3578 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3579 CaMgO_full_tmp_3579 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3580 CaMgO_full_tmp_3580 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3581 CaMgO_full_tmp_3581 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3582 CaMgO_full_tmp_3582 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3583 CaMgO_full_tmp_3583 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3584 CaMgO_full_tmp_3584 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3585 CaMgO_full_tmp_3585 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3586 CaMgO_full_tmp_3586 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3587 CaMgO_full_tmp_3587 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3588 CaMgO_full_tmp_3588 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3589 CaMgO_full_tmp_3589 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3590 CaMgO_full_tmp_3590 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3591 CaMgO_full_tmp_3591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3592 CaMgO_full_tmp_3592 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3593 CaMgO_full_tmp_3593 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3594 CaMgO_full_tmp_3594 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3595 CaMgO_full_tmp_3595 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3596 CaMgO_full_tmp_3596 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3597 CaMgO_full_tmp_3597 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3598 CaMgO_full_tmp_3598 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3599 CaMgO_full_tmp_3599 &

wait
mv CaMgO_full_tmp_3500.f9 CaMgO_full_3500.f9
mv CaMgO_full_tmp_3500.gui CaMgO_full_3500.gui
mv CaMgO_full_tmp_3501.f9 CaMgO_full_3501.f9
mv CaMgO_full_tmp_3501.gui CaMgO_full_3501.gui
mv CaMgO_full_tmp_3502.f9 CaMgO_full_3502.f9
mv CaMgO_full_tmp_3502.gui CaMgO_full_3502.gui
mv CaMgO_full_tmp_3503.f9 CaMgO_full_3503.f9
mv CaMgO_full_tmp_3503.gui CaMgO_full_3503.gui
mv CaMgO_full_tmp_3504.f9 CaMgO_full_3504.f9
mv CaMgO_full_tmp_3504.gui CaMgO_full_3504.gui
mv CaMgO_full_tmp_3505.f9 CaMgO_full_3505.f9
mv CaMgO_full_tmp_3505.gui CaMgO_full_3505.gui
mv CaMgO_full_tmp_3506.f9 CaMgO_full_3506.f9
mv CaMgO_full_tmp_3506.gui CaMgO_full_3506.gui
mv CaMgO_full_tmp_3507.f9 CaMgO_full_3507.f9
mv CaMgO_full_tmp_3507.gui CaMgO_full_3507.gui
mv CaMgO_full_tmp_3508.f9 CaMgO_full_3508.f9
mv CaMgO_full_tmp_3508.gui CaMgO_full_3508.gui
mv CaMgO_full_tmp_3509.f9 CaMgO_full_3509.f9
mv CaMgO_full_tmp_3509.gui CaMgO_full_3509.gui
mv CaMgO_full_tmp_3510.f9 CaMgO_full_3510.f9
mv CaMgO_full_tmp_3510.gui CaMgO_full_3510.gui
mv CaMgO_full_tmp_3511.f9 CaMgO_full_3511.f9
mv CaMgO_full_tmp_3511.gui CaMgO_full_3511.gui
mv CaMgO_full_tmp_3512.f9 CaMgO_full_3512.f9
mv CaMgO_full_tmp_3512.gui CaMgO_full_3512.gui
mv CaMgO_full_tmp_3513.f9 CaMgO_full_3513.f9
mv CaMgO_full_tmp_3513.gui CaMgO_full_3513.gui
mv CaMgO_full_tmp_3514.f9 CaMgO_full_3514.f9
mv CaMgO_full_tmp_3514.gui CaMgO_full_3514.gui
mv CaMgO_full_tmp_3515.f9 CaMgO_full_3515.f9
mv CaMgO_full_tmp_3515.gui CaMgO_full_3515.gui
mv CaMgO_full_tmp_3516.f9 CaMgO_full_3516.f9
mv CaMgO_full_tmp_3516.gui CaMgO_full_3516.gui
mv CaMgO_full_tmp_3517.f9 CaMgO_full_3517.f9
mv CaMgO_full_tmp_3517.gui CaMgO_full_3517.gui
mv CaMgO_full_tmp_3518.f9 CaMgO_full_3518.f9
mv CaMgO_full_tmp_3518.gui CaMgO_full_3518.gui
mv CaMgO_full_tmp_3519.f9 CaMgO_full_3519.f9
mv CaMgO_full_tmp_3519.gui CaMgO_full_3519.gui
mv CaMgO_full_tmp_3520.f9 CaMgO_full_3520.f9
mv CaMgO_full_tmp_3520.gui CaMgO_full_3520.gui
mv CaMgO_full_tmp_3521.f9 CaMgO_full_3521.f9
mv CaMgO_full_tmp_3521.gui CaMgO_full_3521.gui
mv CaMgO_full_tmp_3522.f9 CaMgO_full_3522.f9
mv CaMgO_full_tmp_3522.gui CaMgO_full_3522.gui
mv CaMgO_full_tmp_3523.f9 CaMgO_full_3523.f9
mv CaMgO_full_tmp_3523.gui CaMgO_full_3523.gui
mv CaMgO_full_tmp_3524.f9 CaMgO_full_3524.f9
mv CaMgO_full_tmp_3524.gui CaMgO_full_3524.gui
mv CaMgO_full_tmp_3525.f9 CaMgO_full_3525.f9
mv CaMgO_full_tmp_3525.gui CaMgO_full_3525.gui
mv CaMgO_full_tmp_3526.f9 CaMgO_full_3526.f9
mv CaMgO_full_tmp_3526.gui CaMgO_full_3526.gui
mv CaMgO_full_tmp_3527.f9 CaMgO_full_3527.f9
mv CaMgO_full_tmp_3527.gui CaMgO_full_3527.gui
mv CaMgO_full_tmp_3528.f9 CaMgO_full_3528.f9
mv CaMgO_full_tmp_3528.gui CaMgO_full_3528.gui
mv CaMgO_full_tmp_3529.f9 CaMgO_full_3529.f9
mv CaMgO_full_tmp_3529.gui CaMgO_full_3529.gui
mv CaMgO_full_tmp_3530.f9 CaMgO_full_3530.f9
mv CaMgO_full_tmp_3530.gui CaMgO_full_3530.gui
mv CaMgO_full_tmp_3531.f9 CaMgO_full_3531.f9
mv CaMgO_full_tmp_3531.gui CaMgO_full_3531.gui
mv CaMgO_full_tmp_3532.f9 CaMgO_full_3532.f9
mv CaMgO_full_tmp_3532.gui CaMgO_full_3532.gui
mv CaMgO_full_tmp_3533.f9 CaMgO_full_3533.f9
mv CaMgO_full_tmp_3533.gui CaMgO_full_3533.gui
mv CaMgO_full_tmp_3534.f9 CaMgO_full_3534.f9
mv CaMgO_full_tmp_3534.gui CaMgO_full_3534.gui
mv CaMgO_full_tmp_3535.f9 CaMgO_full_3535.f9
mv CaMgO_full_tmp_3535.gui CaMgO_full_3535.gui
mv CaMgO_full_tmp_3536.f9 CaMgO_full_3536.f9
mv CaMgO_full_tmp_3536.gui CaMgO_full_3536.gui
mv CaMgO_full_tmp_3537.f9 CaMgO_full_3537.f9
mv CaMgO_full_tmp_3537.gui CaMgO_full_3537.gui
mv CaMgO_full_tmp_3538.f9 CaMgO_full_3538.f9
mv CaMgO_full_tmp_3538.gui CaMgO_full_3538.gui
mv CaMgO_full_tmp_3539.f9 CaMgO_full_3539.f9
mv CaMgO_full_tmp_3539.gui CaMgO_full_3539.gui
mv CaMgO_full_tmp_3540.f9 CaMgO_full_3540.f9
mv CaMgO_full_tmp_3540.gui CaMgO_full_3540.gui
mv CaMgO_full_tmp_3541.f9 CaMgO_full_3541.f9
mv CaMgO_full_tmp_3541.gui CaMgO_full_3541.gui
mv CaMgO_full_tmp_3542.f9 CaMgO_full_3542.f9
mv CaMgO_full_tmp_3542.gui CaMgO_full_3542.gui
mv CaMgO_full_tmp_3543.f9 CaMgO_full_3543.f9
mv CaMgO_full_tmp_3543.gui CaMgO_full_3543.gui
mv CaMgO_full_tmp_3544.f9 CaMgO_full_3544.f9
mv CaMgO_full_tmp_3544.gui CaMgO_full_3544.gui
mv CaMgO_full_tmp_3545.f9 CaMgO_full_3545.f9
mv CaMgO_full_tmp_3545.gui CaMgO_full_3545.gui
mv CaMgO_full_tmp_3546.f9 CaMgO_full_3546.f9
mv CaMgO_full_tmp_3546.gui CaMgO_full_3546.gui
mv CaMgO_full_tmp_3547.f9 CaMgO_full_3547.f9
mv CaMgO_full_tmp_3547.gui CaMgO_full_3547.gui
mv CaMgO_full_tmp_3548.f9 CaMgO_full_3548.f9
mv CaMgO_full_tmp_3548.gui CaMgO_full_3548.gui
mv CaMgO_full_tmp_3549.f9 CaMgO_full_3549.f9
mv CaMgO_full_tmp_3549.gui CaMgO_full_3549.gui
mv CaMgO_full_tmp_3550.f9 CaMgO_full_3550.f9
mv CaMgO_full_tmp_3550.gui CaMgO_full_3550.gui
mv CaMgO_full_tmp_3551.f9 CaMgO_full_3551.f9
mv CaMgO_full_tmp_3551.gui CaMgO_full_3551.gui
mv CaMgO_full_tmp_3552.f9 CaMgO_full_3552.f9
mv CaMgO_full_tmp_3552.gui CaMgO_full_3552.gui
mv CaMgO_full_tmp_3553.f9 CaMgO_full_3553.f9
mv CaMgO_full_tmp_3553.gui CaMgO_full_3553.gui
mv CaMgO_full_tmp_3554.f9 CaMgO_full_3554.f9
mv CaMgO_full_tmp_3554.gui CaMgO_full_3554.gui
mv CaMgO_full_tmp_3555.f9 CaMgO_full_3555.f9
mv CaMgO_full_tmp_3555.gui CaMgO_full_3555.gui
mv CaMgO_full_tmp_3556.f9 CaMgO_full_3556.f9
mv CaMgO_full_tmp_3556.gui CaMgO_full_3556.gui
mv CaMgO_full_tmp_3557.f9 CaMgO_full_3557.f9
mv CaMgO_full_tmp_3557.gui CaMgO_full_3557.gui
mv CaMgO_full_tmp_3558.f9 CaMgO_full_3558.f9
mv CaMgO_full_tmp_3558.gui CaMgO_full_3558.gui
mv CaMgO_full_tmp_3559.f9 CaMgO_full_3559.f9
mv CaMgO_full_tmp_3559.gui CaMgO_full_3559.gui
mv CaMgO_full_tmp_3560.f9 CaMgO_full_3560.f9
mv CaMgO_full_tmp_3560.gui CaMgO_full_3560.gui
mv CaMgO_full_tmp_3561.f9 CaMgO_full_3561.f9
mv CaMgO_full_tmp_3561.gui CaMgO_full_3561.gui
mv CaMgO_full_tmp_3562.f9 CaMgO_full_3562.f9
mv CaMgO_full_tmp_3562.gui CaMgO_full_3562.gui
mv CaMgO_full_tmp_3563.f9 CaMgO_full_3563.f9
mv CaMgO_full_tmp_3563.gui CaMgO_full_3563.gui
mv CaMgO_full_tmp_3564.f9 CaMgO_full_3564.f9
mv CaMgO_full_tmp_3564.gui CaMgO_full_3564.gui
mv CaMgO_full_tmp_3565.f9 CaMgO_full_3565.f9
mv CaMgO_full_tmp_3565.gui CaMgO_full_3565.gui
mv CaMgO_full_tmp_3566.f9 CaMgO_full_3566.f9
mv CaMgO_full_tmp_3566.gui CaMgO_full_3566.gui
mv CaMgO_full_tmp_3567.f9 CaMgO_full_3567.f9
mv CaMgO_full_tmp_3567.gui CaMgO_full_3567.gui
mv CaMgO_full_tmp_3568.f9 CaMgO_full_3568.f9
mv CaMgO_full_tmp_3568.gui CaMgO_full_3568.gui
mv CaMgO_full_tmp_3569.f9 CaMgO_full_3569.f9
mv CaMgO_full_tmp_3569.gui CaMgO_full_3569.gui
mv CaMgO_full_tmp_3570.f9 CaMgO_full_3570.f9
mv CaMgO_full_tmp_3570.gui CaMgO_full_3570.gui
mv CaMgO_full_tmp_3571.f9 CaMgO_full_3571.f9
mv CaMgO_full_tmp_3571.gui CaMgO_full_3571.gui
mv CaMgO_full_tmp_3572.f9 CaMgO_full_3572.f9
mv CaMgO_full_tmp_3572.gui CaMgO_full_3572.gui
mv CaMgO_full_tmp_3573.f9 CaMgO_full_3573.f9
mv CaMgO_full_tmp_3573.gui CaMgO_full_3573.gui
mv CaMgO_full_tmp_3574.f9 CaMgO_full_3574.f9
mv CaMgO_full_tmp_3574.gui CaMgO_full_3574.gui
mv CaMgO_full_tmp_3575.f9 CaMgO_full_3575.f9
mv CaMgO_full_tmp_3575.gui CaMgO_full_3575.gui
mv CaMgO_full_tmp_3576.f9 CaMgO_full_3576.f9
mv CaMgO_full_tmp_3576.gui CaMgO_full_3576.gui
mv CaMgO_full_tmp_3577.f9 CaMgO_full_3577.f9
mv CaMgO_full_tmp_3577.gui CaMgO_full_3577.gui
mv CaMgO_full_tmp_3578.f9 CaMgO_full_3578.f9
mv CaMgO_full_tmp_3578.gui CaMgO_full_3578.gui
mv CaMgO_full_tmp_3579.f9 CaMgO_full_3579.f9
mv CaMgO_full_tmp_3579.gui CaMgO_full_3579.gui
mv CaMgO_full_tmp_3580.f9 CaMgO_full_3580.f9
mv CaMgO_full_tmp_3580.gui CaMgO_full_3580.gui
mv CaMgO_full_tmp_3581.f9 CaMgO_full_3581.f9
mv CaMgO_full_tmp_3581.gui CaMgO_full_3581.gui
mv CaMgO_full_tmp_3582.f9 CaMgO_full_3582.f9
mv CaMgO_full_tmp_3582.gui CaMgO_full_3582.gui
mv CaMgO_full_tmp_3583.f9 CaMgO_full_3583.f9
mv CaMgO_full_tmp_3583.gui CaMgO_full_3583.gui
mv CaMgO_full_tmp_3584.f9 CaMgO_full_3584.f9
mv CaMgO_full_tmp_3584.gui CaMgO_full_3584.gui
mv CaMgO_full_tmp_3585.f9 CaMgO_full_3585.f9
mv CaMgO_full_tmp_3585.gui CaMgO_full_3585.gui
mv CaMgO_full_tmp_3586.f9 CaMgO_full_3586.f9
mv CaMgO_full_tmp_3586.gui CaMgO_full_3586.gui
mv CaMgO_full_tmp_3587.f9 CaMgO_full_3587.f9
mv CaMgO_full_tmp_3587.gui CaMgO_full_3587.gui
mv CaMgO_full_tmp_3588.f9 CaMgO_full_3588.f9
mv CaMgO_full_tmp_3588.gui CaMgO_full_3588.gui
mv CaMgO_full_tmp_3589.f9 CaMgO_full_3589.f9
mv CaMgO_full_tmp_3589.gui CaMgO_full_3589.gui
mv CaMgO_full_tmp_3590.f9 CaMgO_full_3590.f9
mv CaMgO_full_tmp_3590.gui CaMgO_full_3590.gui
mv CaMgO_full_tmp_3591.f9 CaMgO_full_3591.f9
mv CaMgO_full_tmp_3591.gui CaMgO_full_3591.gui
mv CaMgO_full_tmp_3592.f9 CaMgO_full_3592.f9
mv CaMgO_full_tmp_3592.gui CaMgO_full_3592.gui
mv CaMgO_full_tmp_3593.f9 CaMgO_full_3593.f9
mv CaMgO_full_tmp_3593.gui CaMgO_full_3593.gui
mv CaMgO_full_tmp_3594.f9 CaMgO_full_3594.f9
mv CaMgO_full_tmp_3594.gui CaMgO_full_3594.gui
mv CaMgO_full_tmp_3595.f9 CaMgO_full_3595.f9
mv CaMgO_full_tmp_3595.gui CaMgO_full_3595.gui
mv CaMgO_full_tmp_3596.f9 CaMgO_full_3596.f9
mv CaMgO_full_tmp_3596.gui CaMgO_full_3596.gui
mv CaMgO_full_tmp_3597.f9 CaMgO_full_3597.f9
mv CaMgO_full_tmp_3597.gui CaMgO_full_3597.gui
mv CaMgO_full_tmp_3598.f9 CaMgO_full_3598.f9
mv CaMgO_full_tmp_3598.gui CaMgO_full_3598.gui
mv CaMgO_full_tmp_3599.f9 CaMgO_full_3599.f9
mv CaMgO_full_tmp_3599.gui CaMgO_full_3599.gui
