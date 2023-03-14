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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3701
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3701


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3702
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3702


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3703
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3703


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3704
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3704


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3705
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3705


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3706
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3706


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3707
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3707


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3708
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3708


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3709
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3709


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3710
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3710


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3711
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3711


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3712
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3712


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3713
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3713


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3714
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3714


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3715
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3715


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3716
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3716


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3717
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3717


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3718
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3718


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3719
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3719


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3720
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3720


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3721
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3721


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3722
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3722


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3723
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3723


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3724
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3724


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3725
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3725


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3726
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3726


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3727
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3727


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3728
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3728


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3729
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3729


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3730
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3730


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3731
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3731


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3732
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3732


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3733
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3733


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3734
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3734


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3735
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3735


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3736
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3736


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3737
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3737


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3738
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3738


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3739
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3739


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3740
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3740


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3741
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3741


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3742
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3742


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3743
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3743


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3744
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3744


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3745
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3745


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3746
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3746


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3747
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3747


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3748
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3748


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3749
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3749


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3750
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3750


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3751
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3751


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3752
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3752


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3753
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3753


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3754
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3754


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3755
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3755


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3756
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3756


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3757
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3757


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3758
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3758


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3759
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3759


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3760
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3760


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3761
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3761


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3762
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3762


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3763
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3763


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3764
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3764


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3765
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3765


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3766
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3766


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3767
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3767


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3768
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3768


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3769
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3769


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3770
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3770


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3771
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3771


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3772
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3772


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3773
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3773


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3774
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3774


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3775
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3775


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3776
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3776


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3777
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3777


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3778
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3778


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3779
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3779


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3780
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3780


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3781
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3781


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3782
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3782


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3783
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3783


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3784
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3784


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3785
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3785


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3786
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3786


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3787
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3787


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3788
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3788


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3789
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3789


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3790
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3790


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3791
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3791


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3792
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3792


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3793
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3793


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3794
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3794


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3795
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3795


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3796
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3796


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3797
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3797


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3798
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3798


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3799
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3799


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3800
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3800


