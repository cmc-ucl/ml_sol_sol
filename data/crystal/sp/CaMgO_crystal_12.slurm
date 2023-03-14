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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1101
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1101


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1102
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1102


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1103
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1103


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1104
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1104


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1105
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1105


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1106
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1106


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1107
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1107


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1108
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1108


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1109
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1109


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1110
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1110


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1111
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1111


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1112
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1112


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1113
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1113


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1114
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1114


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1115
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1115


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1116
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1116


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1117
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1117


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1118
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1118


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1119
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1119


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1120
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1120


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1121
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1121


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1122
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1122


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1123
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1123


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1124
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1124


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1125
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1125


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1126
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1126


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1127
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1127


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1128
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1128


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1129
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1129


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1130
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1130


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1131
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1131


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1132
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1132


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1133
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1133


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1134
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1134


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1135
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1135


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1136
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1136


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1137
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1137


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1138
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1138


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1139
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1139


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1140
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1140


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1141
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1141


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1142
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1142


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1143
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1143


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1144
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1144


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1145
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1145


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1146
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1146


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1147
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1147


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1148
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1148


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1149
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1149


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1150
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1150


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1151
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1151


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1152
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1152


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1153
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1153


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1154
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1154


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1155
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1155


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1156
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1156


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1157
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1157


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1158
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1158


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1159
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1159


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1160
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1160


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1161
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1161


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1162
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1162


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1163
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1163


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1164
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1164


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1165
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1165


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1166
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1166


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1167
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1167


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1168
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1168


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1169
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1169


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1170
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1170


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1171
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1171


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1172
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1172


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1173
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1173


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1174
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1174


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1175
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1175


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1176
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1176


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1177
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1177


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1178
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1178


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1179
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1179


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1180
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1180


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1181
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1181


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1182
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1182


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1183
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1183


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1184
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1184


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1185
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1185


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1186
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1186


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1187
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1187


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1188
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1188


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1189
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1189


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1190
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1190


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1191
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1191


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1192
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1192


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1193
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1193


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1194
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1194


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1195
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1195


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1196
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1196


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1197
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1197


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1198
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1198


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1199
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1199


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1200


