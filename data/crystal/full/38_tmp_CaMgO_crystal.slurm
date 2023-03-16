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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3700 CaMgO_full_tmp_3700 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3701 CaMgO_full_tmp_3701 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3702 CaMgO_full_tmp_3702 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3703 CaMgO_full_tmp_3703 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3704 CaMgO_full_tmp_3704 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3705 CaMgO_full_tmp_3705 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3706 CaMgO_full_tmp_3706 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3707 CaMgO_full_tmp_3707 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3708 CaMgO_full_tmp_3708 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3709 CaMgO_full_tmp_3709 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3710 CaMgO_full_tmp_3710 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3711 CaMgO_full_tmp_3711 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3712 CaMgO_full_tmp_3712 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3713 CaMgO_full_tmp_3713 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3714 CaMgO_full_tmp_3714 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3715 CaMgO_full_tmp_3715 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3716 CaMgO_full_tmp_3716 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3717 CaMgO_full_tmp_3717 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3718 CaMgO_full_tmp_3718 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3719 CaMgO_full_tmp_3719 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3720 CaMgO_full_tmp_3720 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3721 CaMgO_full_tmp_3721 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3722 CaMgO_full_tmp_3722 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3723 CaMgO_full_tmp_3723 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3724 CaMgO_full_tmp_3724 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3725 CaMgO_full_tmp_3725 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3726 CaMgO_full_tmp_3726 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3727 CaMgO_full_tmp_3727 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3728 CaMgO_full_tmp_3728 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3729 CaMgO_full_tmp_3729 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3730 CaMgO_full_tmp_3730 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3731 CaMgO_full_tmp_3731 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3732 CaMgO_full_tmp_3732 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3733 CaMgO_full_tmp_3733 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3734 CaMgO_full_tmp_3734 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3735 CaMgO_full_tmp_3735 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3736 CaMgO_full_tmp_3736 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3737 CaMgO_full_tmp_3737 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3738 CaMgO_full_tmp_3738 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3739 CaMgO_full_tmp_3739 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3740 CaMgO_full_tmp_3740 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3741 CaMgO_full_tmp_3741 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3742 CaMgO_full_tmp_3742 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3743 CaMgO_full_tmp_3743 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3744 CaMgO_full_tmp_3744 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3745 CaMgO_full_tmp_3745 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3746 CaMgO_full_tmp_3746 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3747 CaMgO_full_tmp_3747 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3748 CaMgO_full_tmp_3748 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3749 CaMgO_full_tmp_3749 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3750 CaMgO_full_tmp_3750 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3751 CaMgO_full_tmp_3751 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3752 CaMgO_full_tmp_3752 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3753 CaMgO_full_tmp_3753 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3754 CaMgO_full_tmp_3754 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3755 CaMgO_full_tmp_3755 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3756 CaMgO_full_tmp_3756 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3757 CaMgO_full_tmp_3757 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3758 CaMgO_full_tmp_3758 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3759 CaMgO_full_tmp_3759 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3760 CaMgO_full_tmp_3760 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3761 CaMgO_full_tmp_3761 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3762 CaMgO_full_tmp_3762 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3763 CaMgO_full_tmp_3763 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3764 CaMgO_full_tmp_3764 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3765 CaMgO_full_tmp_3765 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3766 CaMgO_full_tmp_3766 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3767 CaMgO_full_tmp_3767 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3768 CaMgO_full_tmp_3768 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3769 CaMgO_full_tmp_3769 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3770 CaMgO_full_tmp_3770 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3771 CaMgO_full_tmp_3771 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3772 CaMgO_full_tmp_3772 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3773 CaMgO_full_tmp_3773 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3774 CaMgO_full_tmp_3774 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3775 CaMgO_full_tmp_3775 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3776 CaMgO_full_tmp_3776 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3777 CaMgO_full_tmp_3777 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3778 CaMgO_full_tmp_3778 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3779 CaMgO_full_tmp_3779 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3780 CaMgO_full_tmp_3780 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3781 CaMgO_full_tmp_3781 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3782 CaMgO_full_tmp_3782 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3783 CaMgO_full_tmp_3783 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3784 CaMgO_full_tmp_3784 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3785 CaMgO_full_tmp_3785 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3786 CaMgO_full_tmp_3786 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3787 CaMgO_full_tmp_3787 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3788 CaMgO_full_tmp_3788 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3789 CaMgO_full_tmp_3789 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3790 CaMgO_full_tmp_3790 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3791 CaMgO_full_tmp_3791 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3792 CaMgO_full_tmp_3792 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3793 CaMgO_full_tmp_3793 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3794 CaMgO_full_tmp_3794 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3795 CaMgO_full_tmp_3795 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3796 CaMgO_full_tmp_3796 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3797 CaMgO_full_tmp_3797 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3798 CaMgO_full_tmp_3798 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3799 CaMgO_full_tmp_3799 &

wait
mv CaMgO_full_tmp_3700.f9 CaMgO_full_3700.f9
mv CaMgO_full_tmp_3700.gui CaMgO_full_3700.gui
mv CaMgO_full_tmp_3701.f9 CaMgO_full_3701.f9
mv CaMgO_full_tmp_3701.gui CaMgO_full_3701.gui
mv CaMgO_full_tmp_3702.f9 CaMgO_full_3702.f9
mv CaMgO_full_tmp_3702.gui CaMgO_full_3702.gui
mv CaMgO_full_tmp_3703.f9 CaMgO_full_3703.f9
mv CaMgO_full_tmp_3703.gui CaMgO_full_3703.gui
mv CaMgO_full_tmp_3704.f9 CaMgO_full_3704.f9
mv CaMgO_full_tmp_3704.gui CaMgO_full_3704.gui
mv CaMgO_full_tmp_3705.f9 CaMgO_full_3705.f9
mv CaMgO_full_tmp_3705.gui CaMgO_full_3705.gui
mv CaMgO_full_tmp_3706.f9 CaMgO_full_3706.f9
mv CaMgO_full_tmp_3706.gui CaMgO_full_3706.gui
mv CaMgO_full_tmp_3707.f9 CaMgO_full_3707.f9
mv CaMgO_full_tmp_3707.gui CaMgO_full_3707.gui
mv CaMgO_full_tmp_3708.f9 CaMgO_full_3708.f9
mv CaMgO_full_tmp_3708.gui CaMgO_full_3708.gui
mv CaMgO_full_tmp_3709.f9 CaMgO_full_3709.f9
mv CaMgO_full_tmp_3709.gui CaMgO_full_3709.gui
mv CaMgO_full_tmp_3710.f9 CaMgO_full_3710.f9
mv CaMgO_full_tmp_3710.gui CaMgO_full_3710.gui
mv CaMgO_full_tmp_3711.f9 CaMgO_full_3711.f9
mv CaMgO_full_tmp_3711.gui CaMgO_full_3711.gui
mv CaMgO_full_tmp_3712.f9 CaMgO_full_3712.f9
mv CaMgO_full_tmp_3712.gui CaMgO_full_3712.gui
mv CaMgO_full_tmp_3713.f9 CaMgO_full_3713.f9
mv CaMgO_full_tmp_3713.gui CaMgO_full_3713.gui
mv CaMgO_full_tmp_3714.f9 CaMgO_full_3714.f9
mv CaMgO_full_tmp_3714.gui CaMgO_full_3714.gui
mv CaMgO_full_tmp_3715.f9 CaMgO_full_3715.f9
mv CaMgO_full_tmp_3715.gui CaMgO_full_3715.gui
mv CaMgO_full_tmp_3716.f9 CaMgO_full_3716.f9
mv CaMgO_full_tmp_3716.gui CaMgO_full_3716.gui
mv CaMgO_full_tmp_3717.f9 CaMgO_full_3717.f9
mv CaMgO_full_tmp_3717.gui CaMgO_full_3717.gui
mv CaMgO_full_tmp_3718.f9 CaMgO_full_3718.f9
mv CaMgO_full_tmp_3718.gui CaMgO_full_3718.gui
mv CaMgO_full_tmp_3719.f9 CaMgO_full_3719.f9
mv CaMgO_full_tmp_3719.gui CaMgO_full_3719.gui
mv CaMgO_full_tmp_3720.f9 CaMgO_full_3720.f9
mv CaMgO_full_tmp_3720.gui CaMgO_full_3720.gui
mv CaMgO_full_tmp_3721.f9 CaMgO_full_3721.f9
mv CaMgO_full_tmp_3721.gui CaMgO_full_3721.gui
mv CaMgO_full_tmp_3722.f9 CaMgO_full_3722.f9
mv CaMgO_full_tmp_3722.gui CaMgO_full_3722.gui
mv CaMgO_full_tmp_3723.f9 CaMgO_full_3723.f9
mv CaMgO_full_tmp_3723.gui CaMgO_full_3723.gui
mv CaMgO_full_tmp_3724.f9 CaMgO_full_3724.f9
mv CaMgO_full_tmp_3724.gui CaMgO_full_3724.gui
mv CaMgO_full_tmp_3725.f9 CaMgO_full_3725.f9
mv CaMgO_full_tmp_3725.gui CaMgO_full_3725.gui
mv CaMgO_full_tmp_3726.f9 CaMgO_full_3726.f9
mv CaMgO_full_tmp_3726.gui CaMgO_full_3726.gui
mv CaMgO_full_tmp_3727.f9 CaMgO_full_3727.f9
mv CaMgO_full_tmp_3727.gui CaMgO_full_3727.gui
mv CaMgO_full_tmp_3728.f9 CaMgO_full_3728.f9
mv CaMgO_full_tmp_3728.gui CaMgO_full_3728.gui
mv CaMgO_full_tmp_3729.f9 CaMgO_full_3729.f9
mv CaMgO_full_tmp_3729.gui CaMgO_full_3729.gui
mv CaMgO_full_tmp_3730.f9 CaMgO_full_3730.f9
mv CaMgO_full_tmp_3730.gui CaMgO_full_3730.gui
mv CaMgO_full_tmp_3731.f9 CaMgO_full_3731.f9
mv CaMgO_full_tmp_3731.gui CaMgO_full_3731.gui
mv CaMgO_full_tmp_3732.f9 CaMgO_full_3732.f9
mv CaMgO_full_tmp_3732.gui CaMgO_full_3732.gui
mv CaMgO_full_tmp_3733.f9 CaMgO_full_3733.f9
mv CaMgO_full_tmp_3733.gui CaMgO_full_3733.gui
mv CaMgO_full_tmp_3734.f9 CaMgO_full_3734.f9
mv CaMgO_full_tmp_3734.gui CaMgO_full_3734.gui
mv CaMgO_full_tmp_3735.f9 CaMgO_full_3735.f9
mv CaMgO_full_tmp_3735.gui CaMgO_full_3735.gui
mv CaMgO_full_tmp_3736.f9 CaMgO_full_3736.f9
mv CaMgO_full_tmp_3736.gui CaMgO_full_3736.gui
mv CaMgO_full_tmp_3737.f9 CaMgO_full_3737.f9
mv CaMgO_full_tmp_3737.gui CaMgO_full_3737.gui
mv CaMgO_full_tmp_3738.f9 CaMgO_full_3738.f9
mv CaMgO_full_tmp_3738.gui CaMgO_full_3738.gui
mv CaMgO_full_tmp_3739.f9 CaMgO_full_3739.f9
mv CaMgO_full_tmp_3739.gui CaMgO_full_3739.gui
mv CaMgO_full_tmp_3740.f9 CaMgO_full_3740.f9
mv CaMgO_full_tmp_3740.gui CaMgO_full_3740.gui
mv CaMgO_full_tmp_3741.f9 CaMgO_full_3741.f9
mv CaMgO_full_tmp_3741.gui CaMgO_full_3741.gui
mv CaMgO_full_tmp_3742.f9 CaMgO_full_3742.f9
mv CaMgO_full_tmp_3742.gui CaMgO_full_3742.gui
mv CaMgO_full_tmp_3743.f9 CaMgO_full_3743.f9
mv CaMgO_full_tmp_3743.gui CaMgO_full_3743.gui
mv CaMgO_full_tmp_3744.f9 CaMgO_full_3744.f9
mv CaMgO_full_tmp_3744.gui CaMgO_full_3744.gui
mv CaMgO_full_tmp_3745.f9 CaMgO_full_3745.f9
mv CaMgO_full_tmp_3745.gui CaMgO_full_3745.gui
mv CaMgO_full_tmp_3746.f9 CaMgO_full_3746.f9
mv CaMgO_full_tmp_3746.gui CaMgO_full_3746.gui
mv CaMgO_full_tmp_3747.f9 CaMgO_full_3747.f9
mv CaMgO_full_tmp_3747.gui CaMgO_full_3747.gui
mv CaMgO_full_tmp_3748.f9 CaMgO_full_3748.f9
mv CaMgO_full_tmp_3748.gui CaMgO_full_3748.gui
mv CaMgO_full_tmp_3749.f9 CaMgO_full_3749.f9
mv CaMgO_full_tmp_3749.gui CaMgO_full_3749.gui
mv CaMgO_full_tmp_3750.f9 CaMgO_full_3750.f9
mv CaMgO_full_tmp_3750.gui CaMgO_full_3750.gui
mv CaMgO_full_tmp_3751.f9 CaMgO_full_3751.f9
mv CaMgO_full_tmp_3751.gui CaMgO_full_3751.gui
mv CaMgO_full_tmp_3752.f9 CaMgO_full_3752.f9
mv CaMgO_full_tmp_3752.gui CaMgO_full_3752.gui
mv CaMgO_full_tmp_3753.f9 CaMgO_full_3753.f9
mv CaMgO_full_tmp_3753.gui CaMgO_full_3753.gui
mv CaMgO_full_tmp_3754.f9 CaMgO_full_3754.f9
mv CaMgO_full_tmp_3754.gui CaMgO_full_3754.gui
mv CaMgO_full_tmp_3755.f9 CaMgO_full_3755.f9
mv CaMgO_full_tmp_3755.gui CaMgO_full_3755.gui
mv CaMgO_full_tmp_3756.f9 CaMgO_full_3756.f9
mv CaMgO_full_tmp_3756.gui CaMgO_full_3756.gui
mv CaMgO_full_tmp_3757.f9 CaMgO_full_3757.f9
mv CaMgO_full_tmp_3757.gui CaMgO_full_3757.gui
mv CaMgO_full_tmp_3758.f9 CaMgO_full_3758.f9
mv CaMgO_full_tmp_3758.gui CaMgO_full_3758.gui
mv CaMgO_full_tmp_3759.f9 CaMgO_full_3759.f9
mv CaMgO_full_tmp_3759.gui CaMgO_full_3759.gui
mv CaMgO_full_tmp_3760.f9 CaMgO_full_3760.f9
mv CaMgO_full_tmp_3760.gui CaMgO_full_3760.gui
mv CaMgO_full_tmp_3761.f9 CaMgO_full_3761.f9
mv CaMgO_full_tmp_3761.gui CaMgO_full_3761.gui
mv CaMgO_full_tmp_3762.f9 CaMgO_full_3762.f9
mv CaMgO_full_tmp_3762.gui CaMgO_full_3762.gui
mv CaMgO_full_tmp_3763.f9 CaMgO_full_3763.f9
mv CaMgO_full_tmp_3763.gui CaMgO_full_3763.gui
mv CaMgO_full_tmp_3764.f9 CaMgO_full_3764.f9
mv CaMgO_full_tmp_3764.gui CaMgO_full_3764.gui
mv CaMgO_full_tmp_3765.f9 CaMgO_full_3765.f9
mv CaMgO_full_tmp_3765.gui CaMgO_full_3765.gui
mv CaMgO_full_tmp_3766.f9 CaMgO_full_3766.f9
mv CaMgO_full_tmp_3766.gui CaMgO_full_3766.gui
mv CaMgO_full_tmp_3767.f9 CaMgO_full_3767.f9
mv CaMgO_full_tmp_3767.gui CaMgO_full_3767.gui
mv CaMgO_full_tmp_3768.f9 CaMgO_full_3768.f9
mv CaMgO_full_tmp_3768.gui CaMgO_full_3768.gui
mv CaMgO_full_tmp_3769.f9 CaMgO_full_3769.f9
mv CaMgO_full_tmp_3769.gui CaMgO_full_3769.gui
mv CaMgO_full_tmp_3770.f9 CaMgO_full_3770.f9
mv CaMgO_full_tmp_3770.gui CaMgO_full_3770.gui
mv CaMgO_full_tmp_3771.f9 CaMgO_full_3771.f9
mv CaMgO_full_tmp_3771.gui CaMgO_full_3771.gui
mv CaMgO_full_tmp_3772.f9 CaMgO_full_3772.f9
mv CaMgO_full_tmp_3772.gui CaMgO_full_3772.gui
mv CaMgO_full_tmp_3773.f9 CaMgO_full_3773.f9
mv CaMgO_full_tmp_3773.gui CaMgO_full_3773.gui
mv CaMgO_full_tmp_3774.f9 CaMgO_full_3774.f9
mv CaMgO_full_tmp_3774.gui CaMgO_full_3774.gui
mv CaMgO_full_tmp_3775.f9 CaMgO_full_3775.f9
mv CaMgO_full_tmp_3775.gui CaMgO_full_3775.gui
mv CaMgO_full_tmp_3776.f9 CaMgO_full_3776.f9
mv CaMgO_full_tmp_3776.gui CaMgO_full_3776.gui
mv CaMgO_full_tmp_3777.f9 CaMgO_full_3777.f9
mv CaMgO_full_tmp_3777.gui CaMgO_full_3777.gui
mv CaMgO_full_tmp_3778.f9 CaMgO_full_3778.f9
mv CaMgO_full_tmp_3778.gui CaMgO_full_3778.gui
mv CaMgO_full_tmp_3779.f9 CaMgO_full_3779.f9
mv CaMgO_full_tmp_3779.gui CaMgO_full_3779.gui
mv CaMgO_full_tmp_3780.f9 CaMgO_full_3780.f9
mv CaMgO_full_tmp_3780.gui CaMgO_full_3780.gui
mv CaMgO_full_tmp_3781.f9 CaMgO_full_3781.f9
mv CaMgO_full_tmp_3781.gui CaMgO_full_3781.gui
mv CaMgO_full_tmp_3782.f9 CaMgO_full_3782.f9
mv CaMgO_full_tmp_3782.gui CaMgO_full_3782.gui
mv CaMgO_full_tmp_3783.f9 CaMgO_full_3783.f9
mv CaMgO_full_tmp_3783.gui CaMgO_full_3783.gui
mv CaMgO_full_tmp_3784.f9 CaMgO_full_3784.f9
mv CaMgO_full_tmp_3784.gui CaMgO_full_3784.gui
mv CaMgO_full_tmp_3785.f9 CaMgO_full_3785.f9
mv CaMgO_full_tmp_3785.gui CaMgO_full_3785.gui
mv CaMgO_full_tmp_3786.f9 CaMgO_full_3786.f9
mv CaMgO_full_tmp_3786.gui CaMgO_full_3786.gui
mv CaMgO_full_tmp_3787.f9 CaMgO_full_3787.f9
mv CaMgO_full_tmp_3787.gui CaMgO_full_3787.gui
mv CaMgO_full_tmp_3788.f9 CaMgO_full_3788.f9
mv CaMgO_full_tmp_3788.gui CaMgO_full_3788.gui
mv CaMgO_full_tmp_3789.f9 CaMgO_full_3789.f9
mv CaMgO_full_tmp_3789.gui CaMgO_full_3789.gui
mv CaMgO_full_tmp_3790.f9 CaMgO_full_3790.f9
mv CaMgO_full_tmp_3790.gui CaMgO_full_3790.gui
mv CaMgO_full_tmp_3791.f9 CaMgO_full_3791.f9
mv CaMgO_full_tmp_3791.gui CaMgO_full_3791.gui
mv CaMgO_full_tmp_3792.f9 CaMgO_full_3792.f9
mv CaMgO_full_tmp_3792.gui CaMgO_full_3792.gui
mv CaMgO_full_tmp_3793.f9 CaMgO_full_3793.f9
mv CaMgO_full_tmp_3793.gui CaMgO_full_3793.gui
mv CaMgO_full_tmp_3794.f9 CaMgO_full_3794.f9
mv CaMgO_full_tmp_3794.gui CaMgO_full_3794.gui
mv CaMgO_full_tmp_3795.f9 CaMgO_full_3795.f9
mv CaMgO_full_tmp_3795.gui CaMgO_full_3795.gui
mv CaMgO_full_tmp_3796.f9 CaMgO_full_3796.f9
mv CaMgO_full_tmp_3796.gui CaMgO_full_3796.gui
mv CaMgO_full_tmp_3797.f9 CaMgO_full_3797.f9
mv CaMgO_full_tmp_3797.gui CaMgO_full_3797.gui
mv CaMgO_full_tmp_3798.f9 CaMgO_full_3798.f9
mv CaMgO_full_tmp_3798.gui CaMgO_full_3798.gui
mv CaMgO_full_tmp_3799.f9 CaMgO_full_3799.f9
mv CaMgO_full_tmp_3799.gui CaMgO_full_3799.gui
