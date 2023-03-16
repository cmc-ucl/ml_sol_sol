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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3900


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3999


