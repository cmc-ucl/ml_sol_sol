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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3901 CaMgO_coord_3901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3902 CaMgO_coord_3902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3903 CaMgO_coord_3903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3904 CaMgO_coord_3904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3905 CaMgO_coord_3905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3906 CaMgO_coord_3906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3907 CaMgO_coord_3907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3908 CaMgO_coord_3908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3909 CaMgO_coord_3909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3910 CaMgO_coord_3910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3911 CaMgO_coord_3911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3912 CaMgO_coord_3912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3913 CaMgO_coord_3913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3914 CaMgO_coord_3914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3915 CaMgO_coord_3915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3916 CaMgO_coord_3916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3917 CaMgO_coord_3917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3918 CaMgO_coord_3918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3919 CaMgO_coord_3919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3920 CaMgO_coord_3920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3921 CaMgO_coord_3921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3922 CaMgO_coord_3922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3923 CaMgO_coord_3923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3924 CaMgO_coord_3924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3925 CaMgO_coord_3925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3926 CaMgO_coord_3926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3927 CaMgO_coord_3927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3928 CaMgO_coord_3928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3929 CaMgO_coord_3929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3930 CaMgO_coord_3930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3931 CaMgO_coord_3931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3932 CaMgO_coord_3932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3933 CaMgO_coord_3933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3934 CaMgO_coord_3934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3935 CaMgO_coord_3935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3936 CaMgO_coord_3936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3937 CaMgO_coord_3937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3938 CaMgO_coord_3938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3939 CaMgO_coord_3939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3940 CaMgO_coord_3940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3941 CaMgO_coord_3941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3942 CaMgO_coord_3942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3943 CaMgO_coord_3943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3944 CaMgO_coord_3944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3945 CaMgO_coord_3945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3946 CaMgO_coord_3946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3947 CaMgO_coord_3947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3948 CaMgO_coord_3948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3949 CaMgO_coord_3949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3950 CaMgO_coord_3950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3951 CaMgO_coord_3951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3952 CaMgO_coord_3952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3953 CaMgO_coord_3953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3954 CaMgO_coord_3954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3955 CaMgO_coord_3955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3956 CaMgO_coord_3956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3957 CaMgO_coord_3957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3958 CaMgO_coord_3958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3959 CaMgO_coord_3959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3960 CaMgO_coord_3960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3961 CaMgO_coord_3961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3962 CaMgO_coord_3962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3963 CaMgO_coord_3963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3964 CaMgO_coord_3964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3965 CaMgO_coord_3965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3966 CaMgO_coord_3966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3967 CaMgO_coord_3967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3968 CaMgO_coord_3968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3969 CaMgO_coord_3969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3970 CaMgO_coord_3970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3971 CaMgO_coord_3971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3972 CaMgO_coord_3972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3973 CaMgO_coord_3973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3974 CaMgO_coord_3974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3975 CaMgO_coord_3975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3976 CaMgO_coord_3976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3977 CaMgO_coord_3977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3978 CaMgO_coord_3978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3979 CaMgO_coord_3979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3980 CaMgO_coord_3980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3981 CaMgO_coord_3981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3982 CaMgO_coord_3982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3983 CaMgO_coord_3983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3984 CaMgO_coord_3984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3985 CaMgO_coord_3985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3986 CaMgO_coord_3986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3987 CaMgO_coord_3987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3988 CaMgO_coord_3988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3989 CaMgO_coord_3989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3990 CaMgO_coord_3990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3991 CaMgO_coord_3991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3992 CaMgO_coord_3992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3993 CaMgO_coord_3993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3994 CaMgO_coord_3994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3995 CaMgO_coord_3995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3996 CaMgO_coord_3996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3997 CaMgO_coord_3997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3998 CaMgO_coord_3998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3999 CaMgO_coord_3999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3999


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4000 CaMgO_coord_4000
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4000


