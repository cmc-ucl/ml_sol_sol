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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1201 CaMgO_coord_1201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1202 CaMgO_coord_1202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1203 CaMgO_coord_1203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1204 CaMgO_coord_1204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1205 CaMgO_coord_1205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1206 CaMgO_coord_1206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1207 CaMgO_coord_1207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1208 CaMgO_coord_1208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1209 CaMgO_coord_1209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1210 CaMgO_coord_1210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1211 CaMgO_coord_1211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1212 CaMgO_coord_1212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1213 CaMgO_coord_1213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1214 CaMgO_coord_1214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1215 CaMgO_coord_1215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1216 CaMgO_coord_1216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1217 CaMgO_coord_1217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1218 CaMgO_coord_1218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1219 CaMgO_coord_1219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1220 CaMgO_coord_1220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1221 CaMgO_coord_1221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1222 CaMgO_coord_1222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1223 CaMgO_coord_1223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1224 CaMgO_coord_1224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1225 CaMgO_coord_1225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1226 CaMgO_coord_1226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1227 CaMgO_coord_1227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1228 CaMgO_coord_1228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1229 CaMgO_coord_1229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1230 CaMgO_coord_1230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1231 CaMgO_coord_1231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1232 CaMgO_coord_1232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1233 CaMgO_coord_1233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1234 CaMgO_coord_1234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1235 CaMgO_coord_1235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1236 CaMgO_coord_1236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1237 CaMgO_coord_1237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1238 CaMgO_coord_1238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1239 CaMgO_coord_1239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1240 CaMgO_coord_1240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1241 CaMgO_coord_1241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1242 CaMgO_coord_1242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1243 CaMgO_coord_1243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1244 CaMgO_coord_1244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1245 CaMgO_coord_1245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1246 CaMgO_coord_1246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1247 CaMgO_coord_1247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1248 CaMgO_coord_1248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1249 CaMgO_coord_1249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1250 CaMgO_coord_1250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1251 CaMgO_coord_1251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1252 CaMgO_coord_1252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1253 CaMgO_coord_1253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1254 CaMgO_coord_1254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1255 CaMgO_coord_1255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1256 CaMgO_coord_1256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1257 CaMgO_coord_1257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1258 CaMgO_coord_1258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1259 CaMgO_coord_1259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1260 CaMgO_coord_1260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1261 CaMgO_coord_1261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1262 CaMgO_coord_1262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1263 CaMgO_coord_1263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1264 CaMgO_coord_1264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1265 CaMgO_coord_1265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1266 CaMgO_coord_1266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1267 CaMgO_coord_1267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1268 CaMgO_coord_1268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1269 CaMgO_coord_1269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1270 CaMgO_coord_1270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1271 CaMgO_coord_1271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1272 CaMgO_coord_1272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1273 CaMgO_coord_1273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1274 CaMgO_coord_1274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1275 CaMgO_coord_1275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1276 CaMgO_coord_1276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1277 CaMgO_coord_1277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1278 CaMgO_coord_1278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1279 CaMgO_coord_1279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1280 CaMgO_coord_1280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1281 CaMgO_coord_1281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1282 CaMgO_coord_1282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1283 CaMgO_coord_1283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1284 CaMgO_coord_1284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1285 CaMgO_coord_1285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1286 CaMgO_coord_1286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1287 CaMgO_coord_1287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1288 CaMgO_coord_1288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1289 CaMgO_coord_1289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1290 CaMgO_coord_1290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1291 CaMgO_coord_1291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1292 CaMgO_coord_1292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1293 CaMgO_coord_1293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1294 CaMgO_coord_1294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1295 CaMgO_coord_1295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1296 CaMgO_coord_1296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1297 CaMgO_coord_1297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1298 CaMgO_coord_1298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1299 CaMgO_coord_1299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1299


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1300 CaMgO_coord_1300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1300


