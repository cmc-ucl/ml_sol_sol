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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3501
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3501


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3502
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3502


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3503
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3503


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3504
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3504


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3505
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3505


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3506
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3506


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3507
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3507


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3508
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3508


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3509
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3509


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3510
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3510


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3511
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3511


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3512
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3512


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3513
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3513


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3514
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3514


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3515
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3515


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3516
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3516


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3517
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3517


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3518
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3518


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3519
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3519


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3520
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3520


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3521
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3521


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3522
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3522


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3523
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3523


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3524
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3524


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3525
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3525


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3526
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3526


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3527
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3527


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3528
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3528


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3529
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3529


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3530
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3530


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3531
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3531


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3532
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3532


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3533
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3533


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3534
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3534


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3535
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3535


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3536
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3536


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3537
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3537


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3538
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3538


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3539
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3539


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3540
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3540


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3541
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3541


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3542
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3542


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3543
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3543


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3544
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3544


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3545
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3545


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3546
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3546


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3547
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3547


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3548
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3548


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3549
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3549


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3550
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3550


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3551
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3551


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3552
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3552


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3553
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3553


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3554
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3554


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3555
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3555


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3556
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3556


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3557
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3557


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3558
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3558


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3559
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3559


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3560
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3560


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3561
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3561


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3562
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3562


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3563
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3563


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3564
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3564


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3565
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3565


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3566
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3566


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3567
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3567


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3568
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3568


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3569
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3569


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3570
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3570


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3571
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3571


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3572
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3572


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3573
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3573


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3574
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3574


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3575
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3575


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3576
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3576


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3577
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3577


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3578
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3578


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3579
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3579


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3580
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3580


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3581
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3581


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3582
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3582


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3583
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3583


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3584
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3584


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3585
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3585


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3586
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3586


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3587
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3587


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3588
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3588


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3589
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3589


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3590
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3590


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3591
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3591


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3592
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3592


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3593
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3593


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3594
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3594


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3595
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3595


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3596
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3596


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3597
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3597


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3598
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3598


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3599
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3599


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3600


