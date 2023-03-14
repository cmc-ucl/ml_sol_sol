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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_901 CaMgO_coord_901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_902 CaMgO_coord_902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_903 CaMgO_coord_903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_904 CaMgO_coord_904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_905 CaMgO_coord_905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_906 CaMgO_coord_906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_907 CaMgO_coord_907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_908 CaMgO_coord_908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_909 CaMgO_coord_909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_910 CaMgO_coord_910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_911 CaMgO_coord_911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_912 CaMgO_coord_912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_913 CaMgO_coord_913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_914 CaMgO_coord_914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_915 CaMgO_coord_915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_916 CaMgO_coord_916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_917 CaMgO_coord_917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_918 CaMgO_coord_918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_919 CaMgO_coord_919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_920 CaMgO_coord_920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_921 CaMgO_coord_921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_922 CaMgO_coord_922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_923 CaMgO_coord_923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_924 CaMgO_coord_924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_925 CaMgO_coord_925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_926 CaMgO_coord_926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_927 CaMgO_coord_927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_928 CaMgO_coord_928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_929 CaMgO_coord_929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_930 CaMgO_coord_930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_931 CaMgO_coord_931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_932 CaMgO_coord_932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_933 CaMgO_coord_933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_934 CaMgO_coord_934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_935 CaMgO_coord_935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_936 CaMgO_coord_936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_937 CaMgO_coord_937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_938 CaMgO_coord_938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_939 CaMgO_coord_939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_940 CaMgO_coord_940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_941 CaMgO_coord_941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_942 CaMgO_coord_942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_943 CaMgO_coord_943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_944 CaMgO_coord_944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_945 CaMgO_coord_945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_946 CaMgO_coord_946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_947 CaMgO_coord_947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_948 CaMgO_coord_948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_949 CaMgO_coord_949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_950 CaMgO_coord_950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_951 CaMgO_coord_951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_952 CaMgO_coord_952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_953 CaMgO_coord_953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_954 CaMgO_coord_954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_955 CaMgO_coord_955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_956 CaMgO_coord_956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_957 CaMgO_coord_957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_958 CaMgO_coord_958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_959 CaMgO_coord_959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_960 CaMgO_coord_960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_961 CaMgO_coord_961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_962 CaMgO_coord_962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_963 CaMgO_coord_963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_964 CaMgO_coord_964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_965 CaMgO_coord_965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_966 CaMgO_coord_966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_967 CaMgO_coord_967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_968 CaMgO_coord_968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_969 CaMgO_coord_969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_970 CaMgO_coord_970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_971 CaMgO_coord_971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_972 CaMgO_coord_972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_973 CaMgO_coord_973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_974 CaMgO_coord_974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_975 CaMgO_coord_975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_976 CaMgO_coord_976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_977 CaMgO_coord_977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_978 CaMgO_coord_978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_979 CaMgO_coord_979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_980 CaMgO_coord_980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_981 CaMgO_coord_981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_982 CaMgO_coord_982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_983 CaMgO_coord_983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_984 CaMgO_coord_984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_985 CaMgO_coord_985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_986 CaMgO_coord_986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_987 CaMgO_coord_987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_988 CaMgO_coord_988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_989 CaMgO_coord_989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_990 CaMgO_coord_990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_991 CaMgO_coord_991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_992 CaMgO_coord_992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_993 CaMgO_coord_993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_994 CaMgO_coord_994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_995 CaMgO_coord_995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_996 CaMgO_coord_996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_997 CaMgO_coord_997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_998 CaMgO_coord_998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_999 CaMgO_coord_999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_999


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1000 CaMgO_coord_1000
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1000


