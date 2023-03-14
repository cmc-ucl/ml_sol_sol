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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3600 CaMgO_full_tmp_3600 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3601 CaMgO_full_tmp_3601 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3602 CaMgO_full_tmp_3602 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3603 CaMgO_full_tmp_3603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3604 CaMgO_full_tmp_3604 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3605 CaMgO_full_tmp_3605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3606 CaMgO_full_tmp_3606 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3607 CaMgO_full_tmp_3607 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3608 CaMgO_full_tmp_3608 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3609 CaMgO_full_tmp_3609 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3610 CaMgO_full_tmp_3610 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3611 CaMgO_full_tmp_3611 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3612 CaMgO_full_tmp_3612 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3613 CaMgO_full_tmp_3613 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3614 CaMgO_full_tmp_3614 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3615 CaMgO_full_tmp_3615 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3616 CaMgO_full_tmp_3616 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3617 CaMgO_full_tmp_3617 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3618 CaMgO_full_tmp_3618 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3619 CaMgO_full_tmp_3619 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3620 CaMgO_full_tmp_3620 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3621 CaMgO_full_tmp_3621 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3622 CaMgO_full_tmp_3622 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3623 CaMgO_full_tmp_3623 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3624 CaMgO_full_tmp_3624 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3625 CaMgO_full_tmp_3625 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3626 CaMgO_full_tmp_3626 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3627 CaMgO_full_tmp_3627 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3628 CaMgO_full_tmp_3628 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3629 CaMgO_full_tmp_3629 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3630 CaMgO_full_tmp_3630 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3631 CaMgO_full_tmp_3631 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3632 CaMgO_full_tmp_3632 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3633 CaMgO_full_tmp_3633 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3634 CaMgO_full_tmp_3634 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3635 CaMgO_full_tmp_3635 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3636 CaMgO_full_tmp_3636 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3637 CaMgO_full_tmp_3637 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3638 CaMgO_full_tmp_3638 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3639 CaMgO_full_tmp_3639 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3640 CaMgO_full_tmp_3640 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3641 CaMgO_full_tmp_3641 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3642 CaMgO_full_tmp_3642 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3643 CaMgO_full_tmp_3643 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3644 CaMgO_full_tmp_3644 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3645 CaMgO_full_tmp_3645 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3646 CaMgO_full_tmp_3646 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3647 CaMgO_full_tmp_3647 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3648 CaMgO_full_tmp_3648 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3649 CaMgO_full_tmp_3649 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3650 CaMgO_full_tmp_3650 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3651 CaMgO_full_tmp_3651 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3652 CaMgO_full_tmp_3652 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3653 CaMgO_full_tmp_3653 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3654 CaMgO_full_tmp_3654 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3655 CaMgO_full_tmp_3655 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3656 CaMgO_full_tmp_3656 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3657 CaMgO_full_tmp_3657 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3658 CaMgO_full_tmp_3658 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3659 CaMgO_full_tmp_3659 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3660 CaMgO_full_tmp_3660 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3661 CaMgO_full_tmp_3661 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3662 CaMgO_full_tmp_3662 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3663 CaMgO_full_tmp_3663 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3664 CaMgO_full_tmp_3664 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3665 CaMgO_full_tmp_3665 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3666 CaMgO_full_tmp_3666 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3667 CaMgO_full_tmp_3667 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3668 CaMgO_full_tmp_3668 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3669 CaMgO_full_tmp_3669 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3670 CaMgO_full_tmp_3670 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3671 CaMgO_full_tmp_3671 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3672 CaMgO_full_tmp_3672 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3673 CaMgO_full_tmp_3673 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3674 CaMgO_full_tmp_3674 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3675 CaMgO_full_tmp_3675 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3676 CaMgO_full_tmp_3676 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3677 CaMgO_full_tmp_3677 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3678 CaMgO_full_tmp_3678 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3679 CaMgO_full_tmp_3679 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3680 CaMgO_full_tmp_3680 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3681 CaMgO_full_tmp_3681 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3682 CaMgO_full_tmp_3682 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3683 CaMgO_full_tmp_3683 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3684 CaMgO_full_tmp_3684 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3685 CaMgO_full_tmp_3685 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3686 CaMgO_full_tmp_3686 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3687 CaMgO_full_tmp_3687 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3688 CaMgO_full_tmp_3688 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3689 CaMgO_full_tmp_3689 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3690 CaMgO_full_tmp_3690 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3691 CaMgO_full_tmp_3691 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3692 CaMgO_full_tmp_3692 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3693 CaMgO_full_tmp_3693 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3694 CaMgO_full_tmp_3694 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3695 CaMgO_full_tmp_3695 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3696 CaMgO_full_tmp_3696 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3697 CaMgO_full_tmp_3697 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3698 CaMgO_full_tmp_3698 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3699 CaMgO_full_tmp_3699 &

wait
mv CaMgO_full_tmp_3600.f9 CaMgO_full_3600.f9
mv CaMgO_full_tmp_3600.gui CaMgO_full_3600.gui
mv CaMgO_full_tmp_3601.f9 CaMgO_full_3601.f9
mv CaMgO_full_tmp_3601.gui CaMgO_full_3601.gui
mv CaMgO_full_tmp_3602.f9 CaMgO_full_3602.f9
mv CaMgO_full_tmp_3602.gui CaMgO_full_3602.gui
mv CaMgO_full_tmp_3603.f9 CaMgO_full_3603.f9
mv CaMgO_full_tmp_3603.gui CaMgO_full_3603.gui
mv CaMgO_full_tmp_3604.f9 CaMgO_full_3604.f9
mv CaMgO_full_tmp_3604.gui CaMgO_full_3604.gui
mv CaMgO_full_tmp_3605.f9 CaMgO_full_3605.f9
mv CaMgO_full_tmp_3605.gui CaMgO_full_3605.gui
mv CaMgO_full_tmp_3606.f9 CaMgO_full_3606.f9
mv CaMgO_full_tmp_3606.gui CaMgO_full_3606.gui
mv CaMgO_full_tmp_3607.f9 CaMgO_full_3607.f9
mv CaMgO_full_tmp_3607.gui CaMgO_full_3607.gui
mv CaMgO_full_tmp_3608.f9 CaMgO_full_3608.f9
mv CaMgO_full_tmp_3608.gui CaMgO_full_3608.gui
mv CaMgO_full_tmp_3609.f9 CaMgO_full_3609.f9
mv CaMgO_full_tmp_3609.gui CaMgO_full_3609.gui
mv CaMgO_full_tmp_3610.f9 CaMgO_full_3610.f9
mv CaMgO_full_tmp_3610.gui CaMgO_full_3610.gui
mv CaMgO_full_tmp_3611.f9 CaMgO_full_3611.f9
mv CaMgO_full_tmp_3611.gui CaMgO_full_3611.gui
mv CaMgO_full_tmp_3612.f9 CaMgO_full_3612.f9
mv CaMgO_full_tmp_3612.gui CaMgO_full_3612.gui
mv CaMgO_full_tmp_3613.f9 CaMgO_full_3613.f9
mv CaMgO_full_tmp_3613.gui CaMgO_full_3613.gui
mv CaMgO_full_tmp_3614.f9 CaMgO_full_3614.f9
mv CaMgO_full_tmp_3614.gui CaMgO_full_3614.gui
mv CaMgO_full_tmp_3615.f9 CaMgO_full_3615.f9
mv CaMgO_full_tmp_3615.gui CaMgO_full_3615.gui
mv CaMgO_full_tmp_3616.f9 CaMgO_full_3616.f9
mv CaMgO_full_tmp_3616.gui CaMgO_full_3616.gui
mv CaMgO_full_tmp_3617.f9 CaMgO_full_3617.f9
mv CaMgO_full_tmp_3617.gui CaMgO_full_3617.gui
mv CaMgO_full_tmp_3618.f9 CaMgO_full_3618.f9
mv CaMgO_full_tmp_3618.gui CaMgO_full_3618.gui
mv CaMgO_full_tmp_3619.f9 CaMgO_full_3619.f9
mv CaMgO_full_tmp_3619.gui CaMgO_full_3619.gui
mv CaMgO_full_tmp_3620.f9 CaMgO_full_3620.f9
mv CaMgO_full_tmp_3620.gui CaMgO_full_3620.gui
mv CaMgO_full_tmp_3621.f9 CaMgO_full_3621.f9
mv CaMgO_full_tmp_3621.gui CaMgO_full_3621.gui
mv CaMgO_full_tmp_3622.f9 CaMgO_full_3622.f9
mv CaMgO_full_tmp_3622.gui CaMgO_full_3622.gui
mv CaMgO_full_tmp_3623.f9 CaMgO_full_3623.f9
mv CaMgO_full_tmp_3623.gui CaMgO_full_3623.gui
mv CaMgO_full_tmp_3624.f9 CaMgO_full_3624.f9
mv CaMgO_full_tmp_3624.gui CaMgO_full_3624.gui
mv CaMgO_full_tmp_3625.f9 CaMgO_full_3625.f9
mv CaMgO_full_tmp_3625.gui CaMgO_full_3625.gui
mv CaMgO_full_tmp_3626.f9 CaMgO_full_3626.f9
mv CaMgO_full_tmp_3626.gui CaMgO_full_3626.gui
mv CaMgO_full_tmp_3627.f9 CaMgO_full_3627.f9
mv CaMgO_full_tmp_3627.gui CaMgO_full_3627.gui
mv CaMgO_full_tmp_3628.f9 CaMgO_full_3628.f9
mv CaMgO_full_tmp_3628.gui CaMgO_full_3628.gui
mv CaMgO_full_tmp_3629.f9 CaMgO_full_3629.f9
mv CaMgO_full_tmp_3629.gui CaMgO_full_3629.gui
mv CaMgO_full_tmp_3630.f9 CaMgO_full_3630.f9
mv CaMgO_full_tmp_3630.gui CaMgO_full_3630.gui
mv CaMgO_full_tmp_3631.f9 CaMgO_full_3631.f9
mv CaMgO_full_tmp_3631.gui CaMgO_full_3631.gui
mv CaMgO_full_tmp_3632.f9 CaMgO_full_3632.f9
mv CaMgO_full_tmp_3632.gui CaMgO_full_3632.gui
mv CaMgO_full_tmp_3633.f9 CaMgO_full_3633.f9
mv CaMgO_full_tmp_3633.gui CaMgO_full_3633.gui
mv CaMgO_full_tmp_3634.f9 CaMgO_full_3634.f9
mv CaMgO_full_tmp_3634.gui CaMgO_full_3634.gui
mv CaMgO_full_tmp_3635.f9 CaMgO_full_3635.f9
mv CaMgO_full_tmp_3635.gui CaMgO_full_3635.gui
mv CaMgO_full_tmp_3636.f9 CaMgO_full_3636.f9
mv CaMgO_full_tmp_3636.gui CaMgO_full_3636.gui
mv CaMgO_full_tmp_3637.f9 CaMgO_full_3637.f9
mv CaMgO_full_tmp_3637.gui CaMgO_full_3637.gui
mv CaMgO_full_tmp_3638.f9 CaMgO_full_3638.f9
mv CaMgO_full_tmp_3638.gui CaMgO_full_3638.gui
mv CaMgO_full_tmp_3639.f9 CaMgO_full_3639.f9
mv CaMgO_full_tmp_3639.gui CaMgO_full_3639.gui
mv CaMgO_full_tmp_3640.f9 CaMgO_full_3640.f9
mv CaMgO_full_tmp_3640.gui CaMgO_full_3640.gui
mv CaMgO_full_tmp_3641.f9 CaMgO_full_3641.f9
mv CaMgO_full_tmp_3641.gui CaMgO_full_3641.gui
mv CaMgO_full_tmp_3642.f9 CaMgO_full_3642.f9
mv CaMgO_full_tmp_3642.gui CaMgO_full_3642.gui
mv CaMgO_full_tmp_3643.f9 CaMgO_full_3643.f9
mv CaMgO_full_tmp_3643.gui CaMgO_full_3643.gui
mv CaMgO_full_tmp_3644.f9 CaMgO_full_3644.f9
mv CaMgO_full_tmp_3644.gui CaMgO_full_3644.gui
mv CaMgO_full_tmp_3645.f9 CaMgO_full_3645.f9
mv CaMgO_full_tmp_3645.gui CaMgO_full_3645.gui
mv CaMgO_full_tmp_3646.f9 CaMgO_full_3646.f9
mv CaMgO_full_tmp_3646.gui CaMgO_full_3646.gui
mv CaMgO_full_tmp_3647.f9 CaMgO_full_3647.f9
mv CaMgO_full_tmp_3647.gui CaMgO_full_3647.gui
mv CaMgO_full_tmp_3648.f9 CaMgO_full_3648.f9
mv CaMgO_full_tmp_3648.gui CaMgO_full_3648.gui
mv CaMgO_full_tmp_3649.f9 CaMgO_full_3649.f9
mv CaMgO_full_tmp_3649.gui CaMgO_full_3649.gui
mv CaMgO_full_tmp_3650.f9 CaMgO_full_3650.f9
mv CaMgO_full_tmp_3650.gui CaMgO_full_3650.gui
mv CaMgO_full_tmp_3651.f9 CaMgO_full_3651.f9
mv CaMgO_full_tmp_3651.gui CaMgO_full_3651.gui
mv CaMgO_full_tmp_3652.f9 CaMgO_full_3652.f9
mv CaMgO_full_tmp_3652.gui CaMgO_full_3652.gui
mv CaMgO_full_tmp_3653.f9 CaMgO_full_3653.f9
mv CaMgO_full_tmp_3653.gui CaMgO_full_3653.gui
mv CaMgO_full_tmp_3654.f9 CaMgO_full_3654.f9
mv CaMgO_full_tmp_3654.gui CaMgO_full_3654.gui
mv CaMgO_full_tmp_3655.f9 CaMgO_full_3655.f9
mv CaMgO_full_tmp_3655.gui CaMgO_full_3655.gui
mv CaMgO_full_tmp_3656.f9 CaMgO_full_3656.f9
mv CaMgO_full_tmp_3656.gui CaMgO_full_3656.gui
mv CaMgO_full_tmp_3657.f9 CaMgO_full_3657.f9
mv CaMgO_full_tmp_3657.gui CaMgO_full_3657.gui
mv CaMgO_full_tmp_3658.f9 CaMgO_full_3658.f9
mv CaMgO_full_tmp_3658.gui CaMgO_full_3658.gui
mv CaMgO_full_tmp_3659.f9 CaMgO_full_3659.f9
mv CaMgO_full_tmp_3659.gui CaMgO_full_3659.gui
mv CaMgO_full_tmp_3660.f9 CaMgO_full_3660.f9
mv CaMgO_full_tmp_3660.gui CaMgO_full_3660.gui
mv CaMgO_full_tmp_3661.f9 CaMgO_full_3661.f9
mv CaMgO_full_tmp_3661.gui CaMgO_full_3661.gui
mv CaMgO_full_tmp_3662.f9 CaMgO_full_3662.f9
mv CaMgO_full_tmp_3662.gui CaMgO_full_3662.gui
mv CaMgO_full_tmp_3663.f9 CaMgO_full_3663.f9
mv CaMgO_full_tmp_3663.gui CaMgO_full_3663.gui
mv CaMgO_full_tmp_3664.f9 CaMgO_full_3664.f9
mv CaMgO_full_tmp_3664.gui CaMgO_full_3664.gui
mv CaMgO_full_tmp_3665.f9 CaMgO_full_3665.f9
mv CaMgO_full_tmp_3665.gui CaMgO_full_3665.gui
mv CaMgO_full_tmp_3666.f9 CaMgO_full_3666.f9
mv CaMgO_full_tmp_3666.gui CaMgO_full_3666.gui
mv CaMgO_full_tmp_3667.f9 CaMgO_full_3667.f9
mv CaMgO_full_tmp_3667.gui CaMgO_full_3667.gui
mv CaMgO_full_tmp_3668.f9 CaMgO_full_3668.f9
mv CaMgO_full_tmp_3668.gui CaMgO_full_3668.gui
mv CaMgO_full_tmp_3669.f9 CaMgO_full_3669.f9
mv CaMgO_full_tmp_3669.gui CaMgO_full_3669.gui
mv CaMgO_full_tmp_3670.f9 CaMgO_full_3670.f9
mv CaMgO_full_tmp_3670.gui CaMgO_full_3670.gui
mv CaMgO_full_tmp_3671.f9 CaMgO_full_3671.f9
mv CaMgO_full_tmp_3671.gui CaMgO_full_3671.gui
mv CaMgO_full_tmp_3672.f9 CaMgO_full_3672.f9
mv CaMgO_full_tmp_3672.gui CaMgO_full_3672.gui
mv CaMgO_full_tmp_3673.f9 CaMgO_full_3673.f9
mv CaMgO_full_tmp_3673.gui CaMgO_full_3673.gui
mv CaMgO_full_tmp_3674.f9 CaMgO_full_3674.f9
mv CaMgO_full_tmp_3674.gui CaMgO_full_3674.gui
mv CaMgO_full_tmp_3675.f9 CaMgO_full_3675.f9
mv CaMgO_full_tmp_3675.gui CaMgO_full_3675.gui
mv CaMgO_full_tmp_3676.f9 CaMgO_full_3676.f9
mv CaMgO_full_tmp_3676.gui CaMgO_full_3676.gui
mv CaMgO_full_tmp_3677.f9 CaMgO_full_3677.f9
mv CaMgO_full_tmp_3677.gui CaMgO_full_3677.gui
mv CaMgO_full_tmp_3678.f9 CaMgO_full_3678.f9
mv CaMgO_full_tmp_3678.gui CaMgO_full_3678.gui
mv CaMgO_full_tmp_3679.f9 CaMgO_full_3679.f9
mv CaMgO_full_tmp_3679.gui CaMgO_full_3679.gui
mv CaMgO_full_tmp_3680.f9 CaMgO_full_3680.f9
mv CaMgO_full_tmp_3680.gui CaMgO_full_3680.gui
mv CaMgO_full_tmp_3681.f9 CaMgO_full_3681.f9
mv CaMgO_full_tmp_3681.gui CaMgO_full_3681.gui
mv CaMgO_full_tmp_3682.f9 CaMgO_full_3682.f9
mv CaMgO_full_tmp_3682.gui CaMgO_full_3682.gui
mv CaMgO_full_tmp_3683.f9 CaMgO_full_3683.f9
mv CaMgO_full_tmp_3683.gui CaMgO_full_3683.gui
mv CaMgO_full_tmp_3684.f9 CaMgO_full_3684.f9
mv CaMgO_full_tmp_3684.gui CaMgO_full_3684.gui
mv CaMgO_full_tmp_3685.f9 CaMgO_full_3685.f9
mv CaMgO_full_tmp_3685.gui CaMgO_full_3685.gui
mv CaMgO_full_tmp_3686.f9 CaMgO_full_3686.f9
mv CaMgO_full_tmp_3686.gui CaMgO_full_3686.gui
mv CaMgO_full_tmp_3687.f9 CaMgO_full_3687.f9
mv CaMgO_full_tmp_3687.gui CaMgO_full_3687.gui
mv CaMgO_full_tmp_3688.f9 CaMgO_full_3688.f9
mv CaMgO_full_tmp_3688.gui CaMgO_full_3688.gui
mv CaMgO_full_tmp_3689.f9 CaMgO_full_3689.f9
mv CaMgO_full_tmp_3689.gui CaMgO_full_3689.gui
mv CaMgO_full_tmp_3690.f9 CaMgO_full_3690.f9
mv CaMgO_full_tmp_3690.gui CaMgO_full_3690.gui
mv CaMgO_full_tmp_3691.f9 CaMgO_full_3691.f9
mv CaMgO_full_tmp_3691.gui CaMgO_full_3691.gui
mv CaMgO_full_tmp_3692.f9 CaMgO_full_3692.f9
mv CaMgO_full_tmp_3692.gui CaMgO_full_3692.gui
mv CaMgO_full_tmp_3693.f9 CaMgO_full_3693.f9
mv CaMgO_full_tmp_3693.gui CaMgO_full_3693.gui
mv CaMgO_full_tmp_3694.f9 CaMgO_full_3694.f9
mv CaMgO_full_tmp_3694.gui CaMgO_full_3694.gui
mv CaMgO_full_tmp_3695.f9 CaMgO_full_3695.f9
mv CaMgO_full_tmp_3695.gui CaMgO_full_3695.gui
mv CaMgO_full_tmp_3696.f9 CaMgO_full_3696.f9
mv CaMgO_full_tmp_3696.gui CaMgO_full_3696.gui
mv CaMgO_full_tmp_3697.f9 CaMgO_full_3697.f9
mv CaMgO_full_tmp_3697.gui CaMgO_full_3697.gui
mv CaMgO_full_tmp_3698.f9 CaMgO_full_3698.f9
mv CaMgO_full_tmp_3698.gui CaMgO_full_3698.gui
mv CaMgO_full_tmp_3699.f9 CaMgO_full_3699.f9
mv CaMgO_full_tmp_3699.gui CaMgO_full_3699.gui
