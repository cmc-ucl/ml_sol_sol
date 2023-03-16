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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3801 CaMgO_coord_3801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3802 CaMgO_coord_3802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3803 CaMgO_coord_3803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3804 CaMgO_coord_3804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3805 CaMgO_coord_3805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3806 CaMgO_coord_3806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3807 CaMgO_coord_3807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3808 CaMgO_coord_3808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3809 CaMgO_coord_3809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3810 CaMgO_coord_3810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3811 CaMgO_coord_3811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3812 CaMgO_coord_3812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3813 CaMgO_coord_3813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3814 CaMgO_coord_3814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3815 CaMgO_coord_3815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3816 CaMgO_coord_3816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3817 CaMgO_coord_3817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3818 CaMgO_coord_3818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3819 CaMgO_coord_3819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3820 CaMgO_coord_3820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3821 CaMgO_coord_3821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3822 CaMgO_coord_3822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3823 CaMgO_coord_3823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3824 CaMgO_coord_3824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3825 CaMgO_coord_3825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3826 CaMgO_coord_3826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3827 CaMgO_coord_3827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3828 CaMgO_coord_3828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3829 CaMgO_coord_3829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3830 CaMgO_coord_3830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3831 CaMgO_coord_3831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3832 CaMgO_coord_3832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3833 CaMgO_coord_3833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3834 CaMgO_coord_3834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3835 CaMgO_coord_3835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3836 CaMgO_coord_3836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3837 CaMgO_coord_3837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3838 CaMgO_coord_3838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3839 CaMgO_coord_3839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3840 CaMgO_coord_3840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3841 CaMgO_coord_3841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3842 CaMgO_coord_3842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3843 CaMgO_coord_3843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3844 CaMgO_coord_3844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3845 CaMgO_coord_3845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3846 CaMgO_coord_3846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3847 CaMgO_coord_3847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3848 CaMgO_coord_3848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3849 CaMgO_coord_3849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3850 CaMgO_coord_3850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3851 CaMgO_coord_3851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3852 CaMgO_coord_3852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3853 CaMgO_coord_3853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3854 CaMgO_coord_3854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3855 CaMgO_coord_3855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3856 CaMgO_coord_3856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3857 CaMgO_coord_3857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3858 CaMgO_coord_3858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3859 CaMgO_coord_3859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3860 CaMgO_coord_3860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3861 CaMgO_coord_3861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3862 CaMgO_coord_3862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3863 CaMgO_coord_3863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3864 CaMgO_coord_3864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3865 CaMgO_coord_3865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3866 CaMgO_coord_3866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3867 CaMgO_coord_3867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3868 CaMgO_coord_3868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3869 CaMgO_coord_3869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3870 CaMgO_coord_3870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3871 CaMgO_coord_3871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3872 CaMgO_coord_3872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3873 CaMgO_coord_3873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3874 CaMgO_coord_3874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3875 CaMgO_coord_3875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3876 CaMgO_coord_3876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3877 CaMgO_coord_3877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3878 CaMgO_coord_3878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3879 CaMgO_coord_3879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3880 CaMgO_coord_3880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3881 CaMgO_coord_3881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3882 CaMgO_coord_3882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3883 CaMgO_coord_3883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3884 CaMgO_coord_3884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3885 CaMgO_coord_3885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3886 CaMgO_coord_3886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3887 CaMgO_coord_3887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3888 CaMgO_coord_3888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3889 CaMgO_coord_3889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3890 CaMgO_coord_3890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3891 CaMgO_coord_3891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3892 CaMgO_coord_3892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3893 CaMgO_coord_3893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3894 CaMgO_coord_3894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3895 CaMgO_coord_3895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3896 CaMgO_coord_3896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3897 CaMgO_coord_3897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3898 CaMgO_coord_3898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3899 CaMgO_coord_3899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3899


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3900 CaMgO_coord_3900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3900


