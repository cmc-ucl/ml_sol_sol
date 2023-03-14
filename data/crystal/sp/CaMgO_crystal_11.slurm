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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1001
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1001


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1002
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1002


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1003
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1003


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1004
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1004


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1005
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1005


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1006
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1006


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1007
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1007


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1008
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1008


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1009
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1009


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1010
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1010


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1011
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1011


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1012
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1012


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1013
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1013


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1014
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1014


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1015
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1015


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1016
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1016


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1017
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1017


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1018
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1018


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1019
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1019


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1020
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1020


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1021
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1021


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1022
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1022


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1023
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1023


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1024
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1024


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1025
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1025


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1026
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1026


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1027
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1027


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1028
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1028


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1029
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1029


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1030
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1030


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1031
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1031


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1032
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1032


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1033
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1033


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1034
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1034


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1035
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1035


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1036
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1036


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1037
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1037


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1038
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1038


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1039
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1039


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1040
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1040


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1041
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1041


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1042
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1042


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1043
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1043


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1044
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1044


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1045
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1045


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1046
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1046


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1047
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1047


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1048
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1048


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1049
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1049


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1050
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1050


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1051
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1051


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1052
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1052


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1053
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1053


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1054
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1054


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1055
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1055


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1056
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1056


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1057
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1057


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1058
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1058


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1059
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1059


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1060
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1060


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1061
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1061


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1062
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1062


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1063
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1063


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1064
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1064


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1065
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1065


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1066
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1066


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1067
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1067


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1068
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1068


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1069
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1069


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1070
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1070


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1071
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1071


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1072
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1072


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1073
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1073


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1074
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1074


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1075
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1075


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1076
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1076


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1077
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1077


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1078
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1078


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1079
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1079


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1080
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1080


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1081
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1081


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1082
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1082


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1083
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1083


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1084
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1084


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1085
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1085


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1086
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1086


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1087
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1087


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1088
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1088


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1089
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1089


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1090
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1090


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1091
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1091


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1092
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1092


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1093
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1093


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1094
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1094


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1095
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1095


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1096
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1096


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1097
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1097


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1098
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1098


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1099
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1099


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1100
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1100


