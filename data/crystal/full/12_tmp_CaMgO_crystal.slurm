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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1100 CaMgO_full_tmp_1100 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1101 CaMgO_full_tmp_1101 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1102 CaMgO_full_tmp_1102 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1103 CaMgO_full_tmp_1103 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1104 CaMgO_full_tmp_1104 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1105 CaMgO_full_tmp_1105 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1106 CaMgO_full_tmp_1106 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1107 CaMgO_full_tmp_1107 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1108 CaMgO_full_tmp_1108 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1109 CaMgO_full_tmp_1109 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1110 CaMgO_full_tmp_1110 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1111 CaMgO_full_tmp_1111 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1112 CaMgO_full_tmp_1112 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1113 CaMgO_full_tmp_1113 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1114 CaMgO_full_tmp_1114 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1115 CaMgO_full_tmp_1115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1116 CaMgO_full_tmp_1116 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1117 CaMgO_full_tmp_1117 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1118 CaMgO_full_tmp_1118 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1119 CaMgO_full_tmp_1119 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1120 CaMgO_full_tmp_1120 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1121 CaMgO_full_tmp_1121 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1122 CaMgO_full_tmp_1122 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1123 CaMgO_full_tmp_1123 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1124 CaMgO_full_tmp_1124 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1125 CaMgO_full_tmp_1125 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1126 CaMgO_full_tmp_1126 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1127 CaMgO_full_tmp_1127 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1128 CaMgO_full_tmp_1128 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1129 CaMgO_full_tmp_1129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1130 CaMgO_full_tmp_1130 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1131 CaMgO_full_tmp_1131 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1132 CaMgO_full_tmp_1132 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1133 CaMgO_full_tmp_1133 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1134 CaMgO_full_tmp_1134 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1135 CaMgO_full_tmp_1135 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1136 CaMgO_full_tmp_1136 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1137 CaMgO_full_tmp_1137 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1138 CaMgO_full_tmp_1138 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1139 CaMgO_full_tmp_1139 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1140 CaMgO_full_tmp_1140 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1141 CaMgO_full_tmp_1141 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1142 CaMgO_full_tmp_1142 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1143 CaMgO_full_tmp_1143 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1144 CaMgO_full_tmp_1144 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1145 CaMgO_full_tmp_1145 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1146 CaMgO_full_tmp_1146 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1147 CaMgO_full_tmp_1147 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1148 CaMgO_full_tmp_1148 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1149 CaMgO_full_tmp_1149 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1150 CaMgO_full_tmp_1150 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1151 CaMgO_full_tmp_1151 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1152 CaMgO_full_tmp_1152 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1153 CaMgO_full_tmp_1153 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1154 CaMgO_full_tmp_1154 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1155 CaMgO_full_tmp_1155 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1156 CaMgO_full_tmp_1156 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1157 CaMgO_full_tmp_1157 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1158 CaMgO_full_tmp_1158 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1159 CaMgO_full_tmp_1159 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1160 CaMgO_full_tmp_1160 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1161 CaMgO_full_tmp_1161 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1162 CaMgO_full_tmp_1162 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1163 CaMgO_full_tmp_1163 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1164 CaMgO_full_tmp_1164 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1165 CaMgO_full_tmp_1165 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1166 CaMgO_full_tmp_1166 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1167 CaMgO_full_tmp_1167 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1168 CaMgO_full_tmp_1168 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1169 CaMgO_full_tmp_1169 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1170 CaMgO_full_tmp_1170 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1171 CaMgO_full_tmp_1171 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1172 CaMgO_full_tmp_1172 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1173 CaMgO_full_tmp_1173 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1174 CaMgO_full_tmp_1174 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1175 CaMgO_full_tmp_1175 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1176 CaMgO_full_tmp_1176 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1177 CaMgO_full_tmp_1177 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1178 CaMgO_full_tmp_1178 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1179 CaMgO_full_tmp_1179 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1180 CaMgO_full_tmp_1180 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1181 CaMgO_full_tmp_1181 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1182 CaMgO_full_tmp_1182 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1183 CaMgO_full_tmp_1183 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1184 CaMgO_full_tmp_1184 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1185 CaMgO_full_tmp_1185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1186 CaMgO_full_tmp_1186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1187 CaMgO_full_tmp_1187 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1188 CaMgO_full_tmp_1188 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1189 CaMgO_full_tmp_1189 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1190 CaMgO_full_tmp_1190 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1191 CaMgO_full_tmp_1191 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1192 CaMgO_full_tmp_1192 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1193 CaMgO_full_tmp_1193 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1194 CaMgO_full_tmp_1194 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1195 CaMgO_full_tmp_1195 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1196 CaMgO_full_tmp_1196 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1197 CaMgO_full_tmp_1197 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1198 CaMgO_full_tmp_1198 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1199 CaMgO_full_tmp_1199 &

wait
mv CaMgO_full_tmp_1100.f9 CaMgO_full_1100.f9
mv CaMgO_full_tmp_1100.gui CaMgO_full_1100.gui
mv CaMgO_full_tmp_1101.f9 CaMgO_full_1101.f9
mv CaMgO_full_tmp_1101.gui CaMgO_full_1101.gui
mv CaMgO_full_tmp_1102.f9 CaMgO_full_1102.f9
mv CaMgO_full_tmp_1102.gui CaMgO_full_1102.gui
mv CaMgO_full_tmp_1103.f9 CaMgO_full_1103.f9
mv CaMgO_full_tmp_1103.gui CaMgO_full_1103.gui
mv CaMgO_full_tmp_1104.f9 CaMgO_full_1104.f9
mv CaMgO_full_tmp_1104.gui CaMgO_full_1104.gui
mv CaMgO_full_tmp_1105.f9 CaMgO_full_1105.f9
mv CaMgO_full_tmp_1105.gui CaMgO_full_1105.gui
mv CaMgO_full_tmp_1106.f9 CaMgO_full_1106.f9
mv CaMgO_full_tmp_1106.gui CaMgO_full_1106.gui
mv CaMgO_full_tmp_1107.f9 CaMgO_full_1107.f9
mv CaMgO_full_tmp_1107.gui CaMgO_full_1107.gui
mv CaMgO_full_tmp_1108.f9 CaMgO_full_1108.f9
mv CaMgO_full_tmp_1108.gui CaMgO_full_1108.gui
mv CaMgO_full_tmp_1109.f9 CaMgO_full_1109.f9
mv CaMgO_full_tmp_1109.gui CaMgO_full_1109.gui
mv CaMgO_full_tmp_1110.f9 CaMgO_full_1110.f9
mv CaMgO_full_tmp_1110.gui CaMgO_full_1110.gui
mv CaMgO_full_tmp_1111.f9 CaMgO_full_1111.f9
mv CaMgO_full_tmp_1111.gui CaMgO_full_1111.gui
mv CaMgO_full_tmp_1112.f9 CaMgO_full_1112.f9
mv CaMgO_full_tmp_1112.gui CaMgO_full_1112.gui
mv CaMgO_full_tmp_1113.f9 CaMgO_full_1113.f9
mv CaMgO_full_tmp_1113.gui CaMgO_full_1113.gui
mv CaMgO_full_tmp_1114.f9 CaMgO_full_1114.f9
mv CaMgO_full_tmp_1114.gui CaMgO_full_1114.gui
mv CaMgO_full_tmp_1115.f9 CaMgO_full_1115.f9
mv CaMgO_full_tmp_1115.gui CaMgO_full_1115.gui
mv CaMgO_full_tmp_1116.f9 CaMgO_full_1116.f9
mv CaMgO_full_tmp_1116.gui CaMgO_full_1116.gui
mv CaMgO_full_tmp_1117.f9 CaMgO_full_1117.f9
mv CaMgO_full_tmp_1117.gui CaMgO_full_1117.gui
mv CaMgO_full_tmp_1118.f9 CaMgO_full_1118.f9
mv CaMgO_full_tmp_1118.gui CaMgO_full_1118.gui
mv CaMgO_full_tmp_1119.f9 CaMgO_full_1119.f9
mv CaMgO_full_tmp_1119.gui CaMgO_full_1119.gui
mv CaMgO_full_tmp_1120.f9 CaMgO_full_1120.f9
mv CaMgO_full_tmp_1120.gui CaMgO_full_1120.gui
mv CaMgO_full_tmp_1121.f9 CaMgO_full_1121.f9
mv CaMgO_full_tmp_1121.gui CaMgO_full_1121.gui
mv CaMgO_full_tmp_1122.f9 CaMgO_full_1122.f9
mv CaMgO_full_tmp_1122.gui CaMgO_full_1122.gui
mv CaMgO_full_tmp_1123.f9 CaMgO_full_1123.f9
mv CaMgO_full_tmp_1123.gui CaMgO_full_1123.gui
mv CaMgO_full_tmp_1124.f9 CaMgO_full_1124.f9
mv CaMgO_full_tmp_1124.gui CaMgO_full_1124.gui
mv CaMgO_full_tmp_1125.f9 CaMgO_full_1125.f9
mv CaMgO_full_tmp_1125.gui CaMgO_full_1125.gui
mv CaMgO_full_tmp_1126.f9 CaMgO_full_1126.f9
mv CaMgO_full_tmp_1126.gui CaMgO_full_1126.gui
mv CaMgO_full_tmp_1127.f9 CaMgO_full_1127.f9
mv CaMgO_full_tmp_1127.gui CaMgO_full_1127.gui
mv CaMgO_full_tmp_1128.f9 CaMgO_full_1128.f9
mv CaMgO_full_tmp_1128.gui CaMgO_full_1128.gui
mv CaMgO_full_tmp_1129.f9 CaMgO_full_1129.f9
mv CaMgO_full_tmp_1129.gui CaMgO_full_1129.gui
mv CaMgO_full_tmp_1130.f9 CaMgO_full_1130.f9
mv CaMgO_full_tmp_1130.gui CaMgO_full_1130.gui
mv CaMgO_full_tmp_1131.f9 CaMgO_full_1131.f9
mv CaMgO_full_tmp_1131.gui CaMgO_full_1131.gui
mv CaMgO_full_tmp_1132.f9 CaMgO_full_1132.f9
mv CaMgO_full_tmp_1132.gui CaMgO_full_1132.gui
mv CaMgO_full_tmp_1133.f9 CaMgO_full_1133.f9
mv CaMgO_full_tmp_1133.gui CaMgO_full_1133.gui
mv CaMgO_full_tmp_1134.f9 CaMgO_full_1134.f9
mv CaMgO_full_tmp_1134.gui CaMgO_full_1134.gui
mv CaMgO_full_tmp_1135.f9 CaMgO_full_1135.f9
mv CaMgO_full_tmp_1135.gui CaMgO_full_1135.gui
mv CaMgO_full_tmp_1136.f9 CaMgO_full_1136.f9
mv CaMgO_full_tmp_1136.gui CaMgO_full_1136.gui
mv CaMgO_full_tmp_1137.f9 CaMgO_full_1137.f9
mv CaMgO_full_tmp_1137.gui CaMgO_full_1137.gui
mv CaMgO_full_tmp_1138.f9 CaMgO_full_1138.f9
mv CaMgO_full_tmp_1138.gui CaMgO_full_1138.gui
mv CaMgO_full_tmp_1139.f9 CaMgO_full_1139.f9
mv CaMgO_full_tmp_1139.gui CaMgO_full_1139.gui
mv CaMgO_full_tmp_1140.f9 CaMgO_full_1140.f9
mv CaMgO_full_tmp_1140.gui CaMgO_full_1140.gui
mv CaMgO_full_tmp_1141.f9 CaMgO_full_1141.f9
mv CaMgO_full_tmp_1141.gui CaMgO_full_1141.gui
mv CaMgO_full_tmp_1142.f9 CaMgO_full_1142.f9
mv CaMgO_full_tmp_1142.gui CaMgO_full_1142.gui
mv CaMgO_full_tmp_1143.f9 CaMgO_full_1143.f9
mv CaMgO_full_tmp_1143.gui CaMgO_full_1143.gui
mv CaMgO_full_tmp_1144.f9 CaMgO_full_1144.f9
mv CaMgO_full_tmp_1144.gui CaMgO_full_1144.gui
mv CaMgO_full_tmp_1145.f9 CaMgO_full_1145.f9
mv CaMgO_full_tmp_1145.gui CaMgO_full_1145.gui
mv CaMgO_full_tmp_1146.f9 CaMgO_full_1146.f9
mv CaMgO_full_tmp_1146.gui CaMgO_full_1146.gui
mv CaMgO_full_tmp_1147.f9 CaMgO_full_1147.f9
mv CaMgO_full_tmp_1147.gui CaMgO_full_1147.gui
mv CaMgO_full_tmp_1148.f9 CaMgO_full_1148.f9
mv CaMgO_full_tmp_1148.gui CaMgO_full_1148.gui
mv CaMgO_full_tmp_1149.f9 CaMgO_full_1149.f9
mv CaMgO_full_tmp_1149.gui CaMgO_full_1149.gui
mv CaMgO_full_tmp_1150.f9 CaMgO_full_1150.f9
mv CaMgO_full_tmp_1150.gui CaMgO_full_1150.gui
mv CaMgO_full_tmp_1151.f9 CaMgO_full_1151.f9
mv CaMgO_full_tmp_1151.gui CaMgO_full_1151.gui
mv CaMgO_full_tmp_1152.f9 CaMgO_full_1152.f9
mv CaMgO_full_tmp_1152.gui CaMgO_full_1152.gui
mv CaMgO_full_tmp_1153.f9 CaMgO_full_1153.f9
mv CaMgO_full_tmp_1153.gui CaMgO_full_1153.gui
mv CaMgO_full_tmp_1154.f9 CaMgO_full_1154.f9
mv CaMgO_full_tmp_1154.gui CaMgO_full_1154.gui
mv CaMgO_full_tmp_1155.f9 CaMgO_full_1155.f9
mv CaMgO_full_tmp_1155.gui CaMgO_full_1155.gui
mv CaMgO_full_tmp_1156.f9 CaMgO_full_1156.f9
mv CaMgO_full_tmp_1156.gui CaMgO_full_1156.gui
mv CaMgO_full_tmp_1157.f9 CaMgO_full_1157.f9
mv CaMgO_full_tmp_1157.gui CaMgO_full_1157.gui
mv CaMgO_full_tmp_1158.f9 CaMgO_full_1158.f9
mv CaMgO_full_tmp_1158.gui CaMgO_full_1158.gui
mv CaMgO_full_tmp_1159.f9 CaMgO_full_1159.f9
mv CaMgO_full_tmp_1159.gui CaMgO_full_1159.gui
mv CaMgO_full_tmp_1160.f9 CaMgO_full_1160.f9
mv CaMgO_full_tmp_1160.gui CaMgO_full_1160.gui
mv CaMgO_full_tmp_1161.f9 CaMgO_full_1161.f9
mv CaMgO_full_tmp_1161.gui CaMgO_full_1161.gui
mv CaMgO_full_tmp_1162.f9 CaMgO_full_1162.f9
mv CaMgO_full_tmp_1162.gui CaMgO_full_1162.gui
mv CaMgO_full_tmp_1163.f9 CaMgO_full_1163.f9
mv CaMgO_full_tmp_1163.gui CaMgO_full_1163.gui
mv CaMgO_full_tmp_1164.f9 CaMgO_full_1164.f9
mv CaMgO_full_tmp_1164.gui CaMgO_full_1164.gui
mv CaMgO_full_tmp_1165.f9 CaMgO_full_1165.f9
mv CaMgO_full_tmp_1165.gui CaMgO_full_1165.gui
mv CaMgO_full_tmp_1166.f9 CaMgO_full_1166.f9
mv CaMgO_full_tmp_1166.gui CaMgO_full_1166.gui
mv CaMgO_full_tmp_1167.f9 CaMgO_full_1167.f9
mv CaMgO_full_tmp_1167.gui CaMgO_full_1167.gui
mv CaMgO_full_tmp_1168.f9 CaMgO_full_1168.f9
mv CaMgO_full_tmp_1168.gui CaMgO_full_1168.gui
mv CaMgO_full_tmp_1169.f9 CaMgO_full_1169.f9
mv CaMgO_full_tmp_1169.gui CaMgO_full_1169.gui
mv CaMgO_full_tmp_1170.f9 CaMgO_full_1170.f9
mv CaMgO_full_tmp_1170.gui CaMgO_full_1170.gui
mv CaMgO_full_tmp_1171.f9 CaMgO_full_1171.f9
mv CaMgO_full_tmp_1171.gui CaMgO_full_1171.gui
mv CaMgO_full_tmp_1172.f9 CaMgO_full_1172.f9
mv CaMgO_full_tmp_1172.gui CaMgO_full_1172.gui
mv CaMgO_full_tmp_1173.f9 CaMgO_full_1173.f9
mv CaMgO_full_tmp_1173.gui CaMgO_full_1173.gui
mv CaMgO_full_tmp_1174.f9 CaMgO_full_1174.f9
mv CaMgO_full_tmp_1174.gui CaMgO_full_1174.gui
mv CaMgO_full_tmp_1175.f9 CaMgO_full_1175.f9
mv CaMgO_full_tmp_1175.gui CaMgO_full_1175.gui
mv CaMgO_full_tmp_1176.f9 CaMgO_full_1176.f9
mv CaMgO_full_tmp_1176.gui CaMgO_full_1176.gui
mv CaMgO_full_tmp_1177.f9 CaMgO_full_1177.f9
mv CaMgO_full_tmp_1177.gui CaMgO_full_1177.gui
mv CaMgO_full_tmp_1178.f9 CaMgO_full_1178.f9
mv CaMgO_full_tmp_1178.gui CaMgO_full_1178.gui
mv CaMgO_full_tmp_1179.f9 CaMgO_full_1179.f9
mv CaMgO_full_tmp_1179.gui CaMgO_full_1179.gui
mv CaMgO_full_tmp_1180.f9 CaMgO_full_1180.f9
mv CaMgO_full_tmp_1180.gui CaMgO_full_1180.gui
mv CaMgO_full_tmp_1181.f9 CaMgO_full_1181.f9
mv CaMgO_full_tmp_1181.gui CaMgO_full_1181.gui
mv CaMgO_full_tmp_1182.f9 CaMgO_full_1182.f9
mv CaMgO_full_tmp_1182.gui CaMgO_full_1182.gui
mv CaMgO_full_tmp_1183.f9 CaMgO_full_1183.f9
mv CaMgO_full_tmp_1183.gui CaMgO_full_1183.gui
mv CaMgO_full_tmp_1184.f9 CaMgO_full_1184.f9
mv CaMgO_full_tmp_1184.gui CaMgO_full_1184.gui
mv CaMgO_full_tmp_1185.f9 CaMgO_full_1185.f9
mv CaMgO_full_tmp_1185.gui CaMgO_full_1185.gui
mv CaMgO_full_tmp_1186.f9 CaMgO_full_1186.f9
mv CaMgO_full_tmp_1186.gui CaMgO_full_1186.gui
mv CaMgO_full_tmp_1187.f9 CaMgO_full_1187.f9
mv CaMgO_full_tmp_1187.gui CaMgO_full_1187.gui
mv CaMgO_full_tmp_1188.f9 CaMgO_full_1188.f9
mv CaMgO_full_tmp_1188.gui CaMgO_full_1188.gui
mv CaMgO_full_tmp_1189.f9 CaMgO_full_1189.f9
mv CaMgO_full_tmp_1189.gui CaMgO_full_1189.gui
mv CaMgO_full_tmp_1190.f9 CaMgO_full_1190.f9
mv CaMgO_full_tmp_1190.gui CaMgO_full_1190.gui
mv CaMgO_full_tmp_1191.f9 CaMgO_full_1191.f9
mv CaMgO_full_tmp_1191.gui CaMgO_full_1191.gui
mv CaMgO_full_tmp_1192.f9 CaMgO_full_1192.f9
mv CaMgO_full_tmp_1192.gui CaMgO_full_1192.gui
mv CaMgO_full_tmp_1193.f9 CaMgO_full_1193.f9
mv CaMgO_full_tmp_1193.gui CaMgO_full_1193.gui
mv CaMgO_full_tmp_1194.f9 CaMgO_full_1194.f9
mv CaMgO_full_tmp_1194.gui CaMgO_full_1194.gui
mv CaMgO_full_tmp_1195.f9 CaMgO_full_1195.f9
mv CaMgO_full_tmp_1195.gui CaMgO_full_1195.gui
mv CaMgO_full_tmp_1196.f9 CaMgO_full_1196.f9
mv CaMgO_full_tmp_1196.gui CaMgO_full_1196.gui
mv CaMgO_full_tmp_1197.f9 CaMgO_full_1197.f9
mv CaMgO_full_tmp_1197.gui CaMgO_full_1197.gui
mv CaMgO_full_tmp_1198.f9 CaMgO_full_1198.f9
mv CaMgO_full_tmp_1198.gui CaMgO_full_1198.gui
mv CaMgO_full_tmp_1199.f9 CaMgO_full_1199.f9
mv CaMgO_full_tmp_1199.gui CaMgO_full_1199.gui
