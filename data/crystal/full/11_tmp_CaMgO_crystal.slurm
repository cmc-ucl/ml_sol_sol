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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1000 CaMgO_full_tmp_1000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1001 CaMgO_full_tmp_1001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1002 CaMgO_full_tmp_1002 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1003 CaMgO_full_tmp_1003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1004 CaMgO_full_tmp_1004 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1005 CaMgO_full_tmp_1005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1006 CaMgO_full_tmp_1006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1007 CaMgO_full_tmp_1007 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1008 CaMgO_full_tmp_1008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1009 CaMgO_full_tmp_1009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1010 CaMgO_full_tmp_1010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1011 CaMgO_full_tmp_1011 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1012 CaMgO_full_tmp_1012 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1013 CaMgO_full_tmp_1013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1014 CaMgO_full_tmp_1014 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1015 CaMgO_full_tmp_1015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1016 CaMgO_full_tmp_1016 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1017 CaMgO_full_tmp_1017 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1018 CaMgO_full_tmp_1018 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1019 CaMgO_full_tmp_1019 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1020 CaMgO_full_tmp_1020 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1021 CaMgO_full_tmp_1021 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1022 CaMgO_full_tmp_1022 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1023 CaMgO_full_tmp_1023 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1024 CaMgO_full_tmp_1024 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1025 CaMgO_full_tmp_1025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1026 CaMgO_full_tmp_1026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1027 CaMgO_full_tmp_1027 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1028 CaMgO_full_tmp_1028 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1029 CaMgO_full_tmp_1029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1030 CaMgO_full_tmp_1030 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1031 CaMgO_full_tmp_1031 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1032 CaMgO_full_tmp_1032 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1033 CaMgO_full_tmp_1033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1034 CaMgO_full_tmp_1034 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1035 CaMgO_full_tmp_1035 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1036 CaMgO_full_tmp_1036 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1037 CaMgO_full_tmp_1037 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1038 CaMgO_full_tmp_1038 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1039 CaMgO_full_tmp_1039 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1040 CaMgO_full_tmp_1040 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1041 CaMgO_full_tmp_1041 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1042 CaMgO_full_tmp_1042 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1043 CaMgO_full_tmp_1043 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1044 CaMgO_full_tmp_1044 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1045 CaMgO_full_tmp_1045 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1046 CaMgO_full_tmp_1046 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1047 CaMgO_full_tmp_1047 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1048 CaMgO_full_tmp_1048 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1049 CaMgO_full_tmp_1049 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1050 CaMgO_full_tmp_1050 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1051 CaMgO_full_tmp_1051 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1052 CaMgO_full_tmp_1052 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1053 CaMgO_full_tmp_1053 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1054 CaMgO_full_tmp_1054 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1055 CaMgO_full_tmp_1055 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1056 CaMgO_full_tmp_1056 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1057 CaMgO_full_tmp_1057 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1058 CaMgO_full_tmp_1058 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1059 CaMgO_full_tmp_1059 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1060 CaMgO_full_tmp_1060 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1061 CaMgO_full_tmp_1061 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1062 CaMgO_full_tmp_1062 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1063 CaMgO_full_tmp_1063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1064 CaMgO_full_tmp_1064 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1065 CaMgO_full_tmp_1065 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1066 CaMgO_full_tmp_1066 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1067 CaMgO_full_tmp_1067 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1068 CaMgO_full_tmp_1068 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1069 CaMgO_full_tmp_1069 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1070 CaMgO_full_tmp_1070 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1071 CaMgO_full_tmp_1071 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1072 CaMgO_full_tmp_1072 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1073 CaMgO_full_tmp_1073 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1074 CaMgO_full_tmp_1074 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1075 CaMgO_full_tmp_1075 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1076 CaMgO_full_tmp_1076 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1077 CaMgO_full_tmp_1077 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1078 CaMgO_full_tmp_1078 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1079 CaMgO_full_tmp_1079 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1080 CaMgO_full_tmp_1080 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1081 CaMgO_full_tmp_1081 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1082 CaMgO_full_tmp_1082 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1083 CaMgO_full_tmp_1083 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1084 CaMgO_full_tmp_1084 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1085 CaMgO_full_tmp_1085 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1086 CaMgO_full_tmp_1086 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1087 CaMgO_full_tmp_1087 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1088 CaMgO_full_tmp_1088 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1089 CaMgO_full_tmp_1089 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1090 CaMgO_full_tmp_1090 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1091 CaMgO_full_tmp_1091 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1092 CaMgO_full_tmp_1092 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1093 CaMgO_full_tmp_1093 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1094 CaMgO_full_tmp_1094 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1095 CaMgO_full_tmp_1095 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1096 CaMgO_full_tmp_1096 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1097 CaMgO_full_tmp_1097 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1098 CaMgO_full_tmp_1098 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1099 CaMgO_full_tmp_1099 &

wait
mv CaMgO_full_tmp_1000.f9 CaMgO_full_1000.f9
mv CaMgO_full_tmp_1000.gui CaMgO_full_1000.gui
mv CaMgO_full_tmp_1001.f9 CaMgO_full_1001.f9
mv CaMgO_full_tmp_1001.gui CaMgO_full_1001.gui
mv CaMgO_full_tmp_1002.f9 CaMgO_full_1002.f9
mv CaMgO_full_tmp_1002.gui CaMgO_full_1002.gui
mv CaMgO_full_tmp_1003.f9 CaMgO_full_1003.f9
mv CaMgO_full_tmp_1003.gui CaMgO_full_1003.gui
mv CaMgO_full_tmp_1004.f9 CaMgO_full_1004.f9
mv CaMgO_full_tmp_1004.gui CaMgO_full_1004.gui
mv CaMgO_full_tmp_1005.f9 CaMgO_full_1005.f9
mv CaMgO_full_tmp_1005.gui CaMgO_full_1005.gui
mv CaMgO_full_tmp_1006.f9 CaMgO_full_1006.f9
mv CaMgO_full_tmp_1006.gui CaMgO_full_1006.gui
mv CaMgO_full_tmp_1007.f9 CaMgO_full_1007.f9
mv CaMgO_full_tmp_1007.gui CaMgO_full_1007.gui
mv CaMgO_full_tmp_1008.f9 CaMgO_full_1008.f9
mv CaMgO_full_tmp_1008.gui CaMgO_full_1008.gui
mv CaMgO_full_tmp_1009.f9 CaMgO_full_1009.f9
mv CaMgO_full_tmp_1009.gui CaMgO_full_1009.gui
mv CaMgO_full_tmp_1010.f9 CaMgO_full_1010.f9
mv CaMgO_full_tmp_1010.gui CaMgO_full_1010.gui
mv CaMgO_full_tmp_1011.f9 CaMgO_full_1011.f9
mv CaMgO_full_tmp_1011.gui CaMgO_full_1011.gui
mv CaMgO_full_tmp_1012.f9 CaMgO_full_1012.f9
mv CaMgO_full_tmp_1012.gui CaMgO_full_1012.gui
mv CaMgO_full_tmp_1013.f9 CaMgO_full_1013.f9
mv CaMgO_full_tmp_1013.gui CaMgO_full_1013.gui
mv CaMgO_full_tmp_1014.f9 CaMgO_full_1014.f9
mv CaMgO_full_tmp_1014.gui CaMgO_full_1014.gui
mv CaMgO_full_tmp_1015.f9 CaMgO_full_1015.f9
mv CaMgO_full_tmp_1015.gui CaMgO_full_1015.gui
mv CaMgO_full_tmp_1016.f9 CaMgO_full_1016.f9
mv CaMgO_full_tmp_1016.gui CaMgO_full_1016.gui
mv CaMgO_full_tmp_1017.f9 CaMgO_full_1017.f9
mv CaMgO_full_tmp_1017.gui CaMgO_full_1017.gui
mv CaMgO_full_tmp_1018.f9 CaMgO_full_1018.f9
mv CaMgO_full_tmp_1018.gui CaMgO_full_1018.gui
mv CaMgO_full_tmp_1019.f9 CaMgO_full_1019.f9
mv CaMgO_full_tmp_1019.gui CaMgO_full_1019.gui
mv CaMgO_full_tmp_1020.f9 CaMgO_full_1020.f9
mv CaMgO_full_tmp_1020.gui CaMgO_full_1020.gui
mv CaMgO_full_tmp_1021.f9 CaMgO_full_1021.f9
mv CaMgO_full_tmp_1021.gui CaMgO_full_1021.gui
mv CaMgO_full_tmp_1022.f9 CaMgO_full_1022.f9
mv CaMgO_full_tmp_1022.gui CaMgO_full_1022.gui
mv CaMgO_full_tmp_1023.f9 CaMgO_full_1023.f9
mv CaMgO_full_tmp_1023.gui CaMgO_full_1023.gui
mv CaMgO_full_tmp_1024.f9 CaMgO_full_1024.f9
mv CaMgO_full_tmp_1024.gui CaMgO_full_1024.gui
mv CaMgO_full_tmp_1025.f9 CaMgO_full_1025.f9
mv CaMgO_full_tmp_1025.gui CaMgO_full_1025.gui
mv CaMgO_full_tmp_1026.f9 CaMgO_full_1026.f9
mv CaMgO_full_tmp_1026.gui CaMgO_full_1026.gui
mv CaMgO_full_tmp_1027.f9 CaMgO_full_1027.f9
mv CaMgO_full_tmp_1027.gui CaMgO_full_1027.gui
mv CaMgO_full_tmp_1028.f9 CaMgO_full_1028.f9
mv CaMgO_full_tmp_1028.gui CaMgO_full_1028.gui
mv CaMgO_full_tmp_1029.f9 CaMgO_full_1029.f9
mv CaMgO_full_tmp_1029.gui CaMgO_full_1029.gui
mv CaMgO_full_tmp_1030.f9 CaMgO_full_1030.f9
mv CaMgO_full_tmp_1030.gui CaMgO_full_1030.gui
mv CaMgO_full_tmp_1031.f9 CaMgO_full_1031.f9
mv CaMgO_full_tmp_1031.gui CaMgO_full_1031.gui
mv CaMgO_full_tmp_1032.f9 CaMgO_full_1032.f9
mv CaMgO_full_tmp_1032.gui CaMgO_full_1032.gui
mv CaMgO_full_tmp_1033.f9 CaMgO_full_1033.f9
mv CaMgO_full_tmp_1033.gui CaMgO_full_1033.gui
mv CaMgO_full_tmp_1034.f9 CaMgO_full_1034.f9
mv CaMgO_full_tmp_1034.gui CaMgO_full_1034.gui
mv CaMgO_full_tmp_1035.f9 CaMgO_full_1035.f9
mv CaMgO_full_tmp_1035.gui CaMgO_full_1035.gui
mv CaMgO_full_tmp_1036.f9 CaMgO_full_1036.f9
mv CaMgO_full_tmp_1036.gui CaMgO_full_1036.gui
mv CaMgO_full_tmp_1037.f9 CaMgO_full_1037.f9
mv CaMgO_full_tmp_1037.gui CaMgO_full_1037.gui
mv CaMgO_full_tmp_1038.f9 CaMgO_full_1038.f9
mv CaMgO_full_tmp_1038.gui CaMgO_full_1038.gui
mv CaMgO_full_tmp_1039.f9 CaMgO_full_1039.f9
mv CaMgO_full_tmp_1039.gui CaMgO_full_1039.gui
mv CaMgO_full_tmp_1040.f9 CaMgO_full_1040.f9
mv CaMgO_full_tmp_1040.gui CaMgO_full_1040.gui
mv CaMgO_full_tmp_1041.f9 CaMgO_full_1041.f9
mv CaMgO_full_tmp_1041.gui CaMgO_full_1041.gui
mv CaMgO_full_tmp_1042.f9 CaMgO_full_1042.f9
mv CaMgO_full_tmp_1042.gui CaMgO_full_1042.gui
mv CaMgO_full_tmp_1043.f9 CaMgO_full_1043.f9
mv CaMgO_full_tmp_1043.gui CaMgO_full_1043.gui
mv CaMgO_full_tmp_1044.f9 CaMgO_full_1044.f9
mv CaMgO_full_tmp_1044.gui CaMgO_full_1044.gui
mv CaMgO_full_tmp_1045.f9 CaMgO_full_1045.f9
mv CaMgO_full_tmp_1045.gui CaMgO_full_1045.gui
mv CaMgO_full_tmp_1046.f9 CaMgO_full_1046.f9
mv CaMgO_full_tmp_1046.gui CaMgO_full_1046.gui
mv CaMgO_full_tmp_1047.f9 CaMgO_full_1047.f9
mv CaMgO_full_tmp_1047.gui CaMgO_full_1047.gui
mv CaMgO_full_tmp_1048.f9 CaMgO_full_1048.f9
mv CaMgO_full_tmp_1048.gui CaMgO_full_1048.gui
mv CaMgO_full_tmp_1049.f9 CaMgO_full_1049.f9
mv CaMgO_full_tmp_1049.gui CaMgO_full_1049.gui
mv CaMgO_full_tmp_1050.f9 CaMgO_full_1050.f9
mv CaMgO_full_tmp_1050.gui CaMgO_full_1050.gui
mv CaMgO_full_tmp_1051.f9 CaMgO_full_1051.f9
mv CaMgO_full_tmp_1051.gui CaMgO_full_1051.gui
mv CaMgO_full_tmp_1052.f9 CaMgO_full_1052.f9
mv CaMgO_full_tmp_1052.gui CaMgO_full_1052.gui
mv CaMgO_full_tmp_1053.f9 CaMgO_full_1053.f9
mv CaMgO_full_tmp_1053.gui CaMgO_full_1053.gui
mv CaMgO_full_tmp_1054.f9 CaMgO_full_1054.f9
mv CaMgO_full_tmp_1054.gui CaMgO_full_1054.gui
mv CaMgO_full_tmp_1055.f9 CaMgO_full_1055.f9
mv CaMgO_full_tmp_1055.gui CaMgO_full_1055.gui
mv CaMgO_full_tmp_1056.f9 CaMgO_full_1056.f9
mv CaMgO_full_tmp_1056.gui CaMgO_full_1056.gui
mv CaMgO_full_tmp_1057.f9 CaMgO_full_1057.f9
mv CaMgO_full_tmp_1057.gui CaMgO_full_1057.gui
mv CaMgO_full_tmp_1058.f9 CaMgO_full_1058.f9
mv CaMgO_full_tmp_1058.gui CaMgO_full_1058.gui
mv CaMgO_full_tmp_1059.f9 CaMgO_full_1059.f9
mv CaMgO_full_tmp_1059.gui CaMgO_full_1059.gui
mv CaMgO_full_tmp_1060.f9 CaMgO_full_1060.f9
mv CaMgO_full_tmp_1060.gui CaMgO_full_1060.gui
mv CaMgO_full_tmp_1061.f9 CaMgO_full_1061.f9
mv CaMgO_full_tmp_1061.gui CaMgO_full_1061.gui
mv CaMgO_full_tmp_1062.f9 CaMgO_full_1062.f9
mv CaMgO_full_tmp_1062.gui CaMgO_full_1062.gui
mv CaMgO_full_tmp_1063.f9 CaMgO_full_1063.f9
mv CaMgO_full_tmp_1063.gui CaMgO_full_1063.gui
mv CaMgO_full_tmp_1064.f9 CaMgO_full_1064.f9
mv CaMgO_full_tmp_1064.gui CaMgO_full_1064.gui
mv CaMgO_full_tmp_1065.f9 CaMgO_full_1065.f9
mv CaMgO_full_tmp_1065.gui CaMgO_full_1065.gui
mv CaMgO_full_tmp_1066.f9 CaMgO_full_1066.f9
mv CaMgO_full_tmp_1066.gui CaMgO_full_1066.gui
mv CaMgO_full_tmp_1067.f9 CaMgO_full_1067.f9
mv CaMgO_full_tmp_1067.gui CaMgO_full_1067.gui
mv CaMgO_full_tmp_1068.f9 CaMgO_full_1068.f9
mv CaMgO_full_tmp_1068.gui CaMgO_full_1068.gui
mv CaMgO_full_tmp_1069.f9 CaMgO_full_1069.f9
mv CaMgO_full_tmp_1069.gui CaMgO_full_1069.gui
mv CaMgO_full_tmp_1070.f9 CaMgO_full_1070.f9
mv CaMgO_full_tmp_1070.gui CaMgO_full_1070.gui
mv CaMgO_full_tmp_1071.f9 CaMgO_full_1071.f9
mv CaMgO_full_tmp_1071.gui CaMgO_full_1071.gui
mv CaMgO_full_tmp_1072.f9 CaMgO_full_1072.f9
mv CaMgO_full_tmp_1072.gui CaMgO_full_1072.gui
mv CaMgO_full_tmp_1073.f9 CaMgO_full_1073.f9
mv CaMgO_full_tmp_1073.gui CaMgO_full_1073.gui
mv CaMgO_full_tmp_1074.f9 CaMgO_full_1074.f9
mv CaMgO_full_tmp_1074.gui CaMgO_full_1074.gui
mv CaMgO_full_tmp_1075.f9 CaMgO_full_1075.f9
mv CaMgO_full_tmp_1075.gui CaMgO_full_1075.gui
mv CaMgO_full_tmp_1076.f9 CaMgO_full_1076.f9
mv CaMgO_full_tmp_1076.gui CaMgO_full_1076.gui
mv CaMgO_full_tmp_1077.f9 CaMgO_full_1077.f9
mv CaMgO_full_tmp_1077.gui CaMgO_full_1077.gui
mv CaMgO_full_tmp_1078.f9 CaMgO_full_1078.f9
mv CaMgO_full_tmp_1078.gui CaMgO_full_1078.gui
mv CaMgO_full_tmp_1079.f9 CaMgO_full_1079.f9
mv CaMgO_full_tmp_1079.gui CaMgO_full_1079.gui
mv CaMgO_full_tmp_1080.f9 CaMgO_full_1080.f9
mv CaMgO_full_tmp_1080.gui CaMgO_full_1080.gui
mv CaMgO_full_tmp_1081.f9 CaMgO_full_1081.f9
mv CaMgO_full_tmp_1081.gui CaMgO_full_1081.gui
mv CaMgO_full_tmp_1082.f9 CaMgO_full_1082.f9
mv CaMgO_full_tmp_1082.gui CaMgO_full_1082.gui
mv CaMgO_full_tmp_1083.f9 CaMgO_full_1083.f9
mv CaMgO_full_tmp_1083.gui CaMgO_full_1083.gui
mv CaMgO_full_tmp_1084.f9 CaMgO_full_1084.f9
mv CaMgO_full_tmp_1084.gui CaMgO_full_1084.gui
mv CaMgO_full_tmp_1085.f9 CaMgO_full_1085.f9
mv CaMgO_full_tmp_1085.gui CaMgO_full_1085.gui
mv CaMgO_full_tmp_1086.f9 CaMgO_full_1086.f9
mv CaMgO_full_tmp_1086.gui CaMgO_full_1086.gui
mv CaMgO_full_tmp_1087.f9 CaMgO_full_1087.f9
mv CaMgO_full_tmp_1087.gui CaMgO_full_1087.gui
mv CaMgO_full_tmp_1088.f9 CaMgO_full_1088.f9
mv CaMgO_full_tmp_1088.gui CaMgO_full_1088.gui
mv CaMgO_full_tmp_1089.f9 CaMgO_full_1089.f9
mv CaMgO_full_tmp_1089.gui CaMgO_full_1089.gui
mv CaMgO_full_tmp_1090.f9 CaMgO_full_1090.f9
mv CaMgO_full_tmp_1090.gui CaMgO_full_1090.gui
mv CaMgO_full_tmp_1091.f9 CaMgO_full_1091.f9
mv CaMgO_full_tmp_1091.gui CaMgO_full_1091.gui
mv CaMgO_full_tmp_1092.f9 CaMgO_full_1092.f9
mv CaMgO_full_tmp_1092.gui CaMgO_full_1092.gui
mv CaMgO_full_tmp_1093.f9 CaMgO_full_1093.f9
mv CaMgO_full_tmp_1093.gui CaMgO_full_1093.gui
mv CaMgO_full_tmp_1094.f9 CaMgO_full_1094.f9
mv CaMgO_full_tmp_1094.gui CaMgO_full_1094.gui
mv CaMgO_full_tmp_1095.f9 CaMgO_full_1095.f9
mv CaMgO_full_tmp_1095.gui CaMgO_full_1095.gui
mv CaMgO_full_tmp_1096.f9 CaMgO_full_1096.f9
mv CaMgO_full_tmp_1096.gui CaMgO_full_1096.gui
mv CaMgO_full_tmp_1097.f9 CaMgO_full_1097.f9
mv CaMgO_full_tmp_1097.gui CaMgO_full_1097.gui
mv CaMgO_full_tmp_1098.f9 CaMgO_full_1098.f9
mv CaMgO_full_tmp_1098.gui CaMgO_full_1098.gui
mv CaMgO_full_tmp_1099.f9 CaMgO_full_1099.f9
mv CaMgO_full_tmp_1099.gui CaMgO_full_1099.gui
