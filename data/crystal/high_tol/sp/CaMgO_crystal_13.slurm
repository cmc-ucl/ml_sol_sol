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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1200


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1299


