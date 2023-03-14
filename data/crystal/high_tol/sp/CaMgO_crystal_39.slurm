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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3800
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3800


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3899


