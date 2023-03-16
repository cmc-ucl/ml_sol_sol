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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_900


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_999


