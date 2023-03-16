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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1200 CaMgO_full_tmp_1200 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1201 CaMgO_full_tmp_1201 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1202 CaMgO_full_tmp_1202 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1203 CaMgO_full_tmp_1203 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1204 CaMgO_full_tmp_1204 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1205 CaMgO_full_tmp_1205 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1206 CaMgO_full_tmp_1206 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1207 CaMgO_full_tmp_1207 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1208 CaMgO_full_tmp_1208 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1209 CaMgO_full_tmp_1209 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1210 CaMgO_full_tmp_1210 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1211 CaMgO_full_tmp_1211 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1212 CaMgO_full_tmp_1212 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1213 CaMgO_full_tmp_1213 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1214 CaMgO_full_tmp_1214 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1215 CaMgO_full_tmp_1215 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1216 CaMgO_full_tmp_1216 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1217 CaMgO_full_tmp_1217 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1218 CaMgO_full_tmp_1218 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1219 CaMgO_full_tmp_1219 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1220 CaMgO_full_tmp_1220 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1221 CaMgO_full_tmp_1221 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1222 CaMgO_full_tmp_1222 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1223 CaMgO_full_tmp_1223 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1224 CaMgO_full_tmp_1224 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1225 CaMgO_full_tmp_1225 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1226 CaMgO_full_tmp_1226 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1227 CaMgO_full_tmp_1227 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1228 CaMgO_full_tmp_1228 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1229 CaMgO_full_tmp_1229 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1230 CaMgO_full_tmp_1230 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1231 CaMgO_full_tmp_1231 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1232 CaMgO_full_tmp_1232 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1233 CaMgO_full_tmp_1233 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1234 CaMgO_full_tmp_1234 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1235 CaMgO_full_tmp_1235 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1236 CaMgO_full_tmp_1236 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1237 CaMgO_full_tmp_1237 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1238 CaMgO_full_tmp_1238 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1239 CaMgO_full_tmp_1239 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1240 CaMgO_full_tmp_1240 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1241 CaMgO_full_tmp_1241 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1242 CaMgO_full_tmp_1242 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1243 CaMgO_full_tmp_1243 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1244 CaMgO_full_tmp_1244 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1245 CaMgO_full_tmp_1245 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1246 CaMgO_full_tmp_1246 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1247 CaMgO_full_tmp_1247 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1248 CaMgO_full_tmp_1248 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1249 CaMgO_full_tmp_1249 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1250 CaMgO_full_tmp_1250 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1251 CaMgO_full_tmp_1251 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1252 CaMgO_full_tmp_1252 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1253 CaMgO_full_tmp_1253 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1254 CaMgO_full_tmp_1254 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1255 CaMgO_full_tmp_1255 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1256 CaMgO_full_tmp_1256 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1257 CaMgO_full_tmp_1257 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1258 CaMgO_full_tmp_1258 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1259 CaMgO_full_tmp_1259 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1260 CaMgO_full_tmp_1260 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1261 CaMgO_full_tmp_1261 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1262 CaMgO_full_tmp_1262 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1263 CaMgO_full_tmp_1263 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1264 CaMgO_full_tmp_1264 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1265 CaMgO_full_tmp_1265 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1266 CaMgO_full_tmp_1266 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1267 CaMgO_full_tmp_1267 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1268 CaMgO_full_tmp_1268 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1269 CaMgO_full_tmp_1269 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1270 CaMgO_full_tmp_1270 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1271 CaMgO_full_tmp_1271 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1272 CaMgO_full_tmp_1272 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1273 CaMgO_full_tmp_1273 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1274 CaMgO_full_tmp_1274 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1275 CaMgO_full_tmp_1275 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1276 CaMgO_full_tmp_1276 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1277 CaMgO_full_tmp_1277 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1278 CaMgO_full_tmp_1278 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1279 CaMgO_full_tmp_1279 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1280 CaMgO_full_tmp_1280 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1281 CaMgO_full_tmp_1281 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1282 CaMgO_full_tmp_1282 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1283 CaMgO_full_tmp_1283 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1284 CaMgO_full_tmp_1284 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1285 CaMgO_full_tmp_1285 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1286 CaMgO_full_tmp_1286 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1287 CaMgO_full_tmp_1287 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1288 CaMgO_full_tmp_1288 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1289 CaMgO_full_tmp_1289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1290 CaMgO_full_tmp_1290 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1291 CaMgO_full_tmp_1291 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1292 CaMgO_full_tmp_1292 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1293 CaMgO_full_tmp_1293 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1294 CaMgO_full_tmp_1294 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1295 CaMgO_full_tmp_1295 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1296 CaMgO_full_tmp_1296 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1297 CaMgO_full_tmp_1297 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1298 CaMgO_full_tmp_1298 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1299 CaMgO_full_tmp_1299 &

wait
mv CaMgO_full_tmp_1200.f9 CaMgO_full_1200.f9
mv CaMgO_full_tmp_1200.gui CaMgO_full_1200.gui
mv CaMgO_full_tmp_1201.f9 CaMgO_full_1201.f9
mv CaMgO_full_tmp_1201.gui CaMgO_full_1201.gui
mv CaMgO_full_tmp_1202.f9 CaMgO_full_1202.f9
mv CaMgO_full_tmp_1202.gui CaMgO_full_1202.gui
mv CaMgO_full_tmp_1203.f9 CaMgO_full_1203.f9
mv CaMgO_full_tmp_1203.gui CaMgO_full_1203.gui
mv CaMgO_full_tmp_1204.f9 CaMgO_full_1204.f9
mv CaMgO_full_tmp_1204.gui CaMgO_full_1204.gui
mv CaMgO_full_tmp_1205.f9 CaMgO_full_1205.f9
mv CaMgO_full_tmp_1205.gui CaMgO_full_1205.gui
mv CaMgO_full_tmp_1206.f9 CaMgO_full_1206.f9
mv CaMgO_full_tmp_1206.gui CaMgO_full_1206.gui
mv CaMgO_full_tmp_1207.f9 CaMgO_full_1207.f9
mv CaMgO_full_tmp_1207.gui CaMgO_full_1207.gui
mv CaMgO_full_tmp_1208.f9 CaMgO_full_1208.f9
mv CaMgO_full_tmp_1208.gui CaMgO_full_1208.gui
mv CaMgO_full_tmp_1209.f9 CaMgO_full_1209.f9
mv CaMgO_full_tmp_1209.gui CaMgO_full_1209.gui
mv CaMgO_full_tmp_1210.f9 CaMgO_full_1210.f9
mv CaMgO_full_tmp_1210.gui CaMgO_full_1210.gui
mv CaMgO_full_tmp_1211.f9 CaMgO_full_1211.f9
mv CaMgO_full_tmp_1211.gui CaMgO_full_1211.gui
mv CaMgO_full_tmp_1212.f9 CaMgO_full_1212.f9
mv CaMgO_full_tmp_1212.gui CaMgO_full_1212.gui
mv CaMgO_full_tmp_1213.f9 CaMgO_full_1213.f9
mv CaMgO_full_tmp_1213.gui CaMgO_full_1213.gui
mv CaMgO_full_tmp_1214.f9 CaMgO_full_1214.f9
mv CaMgO_full_tmp_1214.gui CaMgO_full_1214.gui
mv CaMgO_full_tmp_1215.f9 CaMgO_full_1215.f9
mv CaMgO_full_tmp_1215.gui CaMgO_full_1215.gui
mv CaMgO_full_tmp_1216.f9 CaMgO_full_1216.f9
mv CaMgO_full_tmp_1216.gui CaMgO_full_1216.gui
mv CaMgO_full_tmp_1217.f9 CaMgO_full_1217.f9
mv CaMgO_full_tmp_1217.gui CaMgO_full_1217.gui
mv CaMgO_full_tmp_1218.f9 CaMgO_full_1218.f9
mv CaMgO_full_tmp_1218.gui CaMgO_full_1218.gui
mv CaMgO_full_tmp_1219.f9 CaMgO_full_1219.f9
mv CaMgO_full_tmp_1219.gui CaMgO_full_1219.gui
mv CaMgO_full_tmp_1220.f9 CaMgO_full_1220.f9
mv CaMgO_full_tmp_1220.gui CaMgO_full_1220.gui
mv CaMgO_full_tmp_1221.f9 CaMgO_full_1221.f9
mv CaMgO_full_tmp_1221.gui CaMgO_full_1221.gui
mv CaMgO_full_tmp_1222.f9 CaMgO_full_1222.f9
mv CaMgO_full_tmp_1222.gui CaMgO_full_1222.gui
mv CaMgO_full_tmp_1223.f9 CaMgO_full_1223.f9
mv CaMgO_full_tmp_1223.gui CaMgO_full_1223.gui
mv CaMgO_full_tmp_1224.f9 CaMgO_full_1224.f9
mv CaMgO_full_tmp_1224.gui CaMgO_full_1224.gui
mv CaMgO_full_tmp_1225.f9 CaMgO_full_1225.f9
mv CaMgO_full_tmp_1225.gui CaMgO_full_1225.gui
mv CaMgO_full_tmp_1226.f9 CaMgO_full_1226.f9
mv CaMgO_full_tmp_1226.gui CaMgO_full_1226.gui
mv CaMgO_full_tmp_1227.f9 CaMgO_full_1227.f9
mv CaMgO_full_tmp_1227.gui CaMgO_full_1227.gui
mv CaMgO_full_tmp_1228.f9 CaMgO_full_1228.f9
mv CaMgO_full_tmp_1228.gui CaMgO_full_1228.gui
mv CaMgO_full_tmp_1229.f9 CaMgO_full_1229.f9
mv CaMgO_full_tmp_1229.gui CaMgO_full_1229.gui
mv CaMgO_full_tmp_1230.f9 CaMgO_full_1230.f9
mv CaMgO_full_tmp_1230.gui CaMgO_full_1230.gui
mv CaMgO_full_tmp_1231.f9 CaMgO_full_1231.f9
mv CaMgO_full_tmp_1231.gui CaMgO_full_1231.gui
mv CaMgO_full_tmp_1232.f9 CaMgO_full_1232.f9
mv CaMgO_full_tmp_1232.gui CaMgO_full_1232.gui
mv CaMgO_full_tmp_1233.f9 CaMgO_full_1233.f9
mv CaMgO_full_tmp_1233.gui CaMgO_full_1233.gui
mv CaMgO_full_tmp_1234.f9 CaMgO_full_1234.f9
mv CaMgO_full_tmp_1234.gui CaMgO_full_1234.gui
mv CaMgO_full_tmp_1235.f9 CaMgO_full_1235.f9
mv CaMgO_full_tmp_1235.gui CaMgO_full_1235.gui
mv CaMgO_full_tmp_1236.f9 CaMgO_full_1236.f9
mv CaMgO_full_tmp_1236.gui CaMgO_full_1236.gui
mv CaMgO_full_tmp_1237.f9 CaMgO_full_1237.f9
mv CaMgO_full_tmp_1237.gui CaMgO_full_1237.gui
mv CaMgO_full_tmp_1238.f9 CaMgO_full_1238.f9
mv CaMgO_full_tmp_1238.gui CaMgO_full_1238.gui
mv CaMgO_full_tmp_1239.f9 CaMgO_full_1239.f9
mv CaMgO_full_tmp_1239.gui CaMgO_full_1239.gui
mv CaMgO_full_tmp_1240.f9 CaMgO_full_1240.f9
mv CaMgO_full_tmp_1240.gui CaMgO_full_1240.gui
mv CaMgO_full_tmp_1241.f9 CaMgO_full_1241.f9
mv CaMgO_full_tmp_1241.gui CaMgO_full_1241.gui
mv CaMgO_full_tmp_1242.f9 CaMgO_full_1242.f9
mv CaMgO_full_tmp_1242.gui CaMgO_full_1242.gui
mv CaMgO_full_tmp_1243.f9 CaMgO_full_1243.f9
mv CaMgO_full_tmp_1243.gui CaMgO_full_1243.gui
mv CaMgO_full_tmp_1244.f9 CaMgO_full_1244.f9
mv CaMgO_full_tmp_1244.gui CaMgO_full_1244.gui
mv CaMgO_full_tmp_1245.f9 CaMgO_full_1245.f9
mv CaMgO_full_tmp_1245.gui CaMgO_full_1245.gui
mv CaMgO_full_tmp_1246.f9 CaMgO_full_1246.f9
mv CaMgO_full_tmp_1246.gui CaMgO_full_1246.gui
mv CaMgO_full_tmp_1247.f9 CaMgO_full_1247.f9
mv CaMgO_full_tmp_1247.gui CaMgO_full_1247.gui
mv CaMgO_full_tmp_1248.f9 CaMgO_full_1248.f9
mv CaMgO_full_tmp_1248.gui CaMgO_full_1248.gui
mv CaMgO_full_tmp_1249.f9 CaMgO_full_1249.f9
mv CaMgO_full_tmp_1249.gui CaMgO_full_1249.gui
mv CaMgO_full_tmp_1250.f9 CaMgO_full_1250.f9
mv CaMgO_full_tmp_1250.gui CaMgO_full_1250.gui
mv CaMgO_full_tmp_1251.f9 CaMgO_full_1251.f9
mv CaMgO_full_tmp_1251.gui CaMgO_full_1251.gui
mv CaMgO_full_tmp_1252.f9 CaMgO_full_1252.f9
mv CaMgO_full_tmp_1252.gui CaMgO_full_1252.gui
mv CaMgO_full_tmp_1253.f9 CaMgO_full_1253.f9
mv CaMgO_full_tmp_1253.gui CaMgO_full_1253.gui
mv CaMgO_full_tmp_1254.f9 CaMgO_full_1254.f9
mv CaMgO_full_tmp_1254.gui CaMgO_full_1254.gui
mv CaMgO_full_tmp_1255.f9 CaMgO_full_1255.f9
mv CaMgO_full_tmp_1255.gui CaMgO_full_1255.gui
mv CaMgO_full_tmp_1256.f9 CaMgO_full_1256.f9
mv CaMgO_full_tmp_1256.gui CaMgO_full_1256.gui
mv CaMgO_full_tmp_1257.f9 CaMgO_full_1257.f9
mv CaMgO_full_tmp_1257.gui CaMgO_full_1257.gui
mv CaMgO_full_tmp_1258.f9 CaMgO_full_1258.f9
mv CaMgO_full_tmp_1258.gui CaMgO_full_1258.gui
mv CaMgO_full_tmp_1259.f9 CaMgO_full_1259.f9
mv CaMgO_full_tmp_1259.gui CaMgO_full_1259.gui
mv CaMgO_full_tmp_1260.f9 CaMgO_full_1260.f9
mv CaMgO_full_tmp_1260.gui CaMgO_full_1260.gui
mv CaMgO_full_tmp_1261.f9 CaMgO_full_1261.f9
mv CaMgO_full_tmp_1261.gui CaMgO_full_1261.gui
mv CaMgO_full_tmp_1262.f9 CaMgO_full_1262.f9
mv CaMgO_full_tmp_1262.gui CaMgO_full_1262.gui
mv CaMgO_full_tmp_1263.f9 CaMgO_full_1263.f9
mv CaMgO_full_tmp_1263.gui CaMgO_full_1263.gui
mv CaMgO_full_tmp_1264.f9 CaMgO_full_1264.f9
mv CaMgO_full_tmp_1264.gui CaMgO_full_1264.gui
mv CaMgO_full_tmp_1265.f9 CaMgO_full_1265.f9
mv CaMgO_full_tmp_1265.gui CaMgO_full_1265.gui
mv CaMgO_full_tmp_1266.f9 CaMgO_full_1266.f9
mv CaMgO_full_tmp_1266.gui CaMgO_full_1266.gui
mv CaMgO_full_tmp_1267.f9 CaMgO_full_1267.f9
mv CaMgO_full_tmp_1267.gui CaMgO_full_1267.gui
mv CaMgO_full_tmp_1268.f9 CaMgO_full_1268.f9
mv CaMgO_full_tmp_1268.gui CaMgO_full_1268.gui
mv CaMgO_full_tmp_1269.f9 CaMgO_full_1269.f9
mv CaMgO_full_tmp_1269.gui CaMgO_full_1269.gui
mv CaMgO_full_tmp_1270.f9 CaMgO_full_1270.f9
mv CaMgO_full_tmp_1270.gui CaMgO_full_1270.gui
mv CaMgO_full_tmp_1271.f9 CaMgO_full_1271.f9
mv CaMgO_full_tmp_1271.gui CaMgO_full_1271.gui
mv CaMgO_full_tmp_1272.f9 CaMgO_full_1272.f9
mv CaMgO_full_tmp_1272.gui CaMgO_full_1272.gui
mv CaMgO_full_tmp_1273.f9 CaMgO_full_1273.f9
mv CaMgO_full_tmp_1273.gui CaMgO_full_1273.gui
mv CaMgO_full_tmp_1274.f9 CaMgO_full_1274.f9
mv CaMgO_full_tmp_1274.gui CaMgO_full_1274.gui
mv CaMgO_full_tmp_1275.f9 CaMgO_full_1275.f9
mv CaMgO_full_tmp_1275.gui CaMgO_full_1275.gui
mv CaMgO_full_tmp_1276.f9 CaMgO_full_1276.f9
mv CaMgO_full_tmp_1276.gui CaMgO_full_1276.gui
mv CaMgO_full_tmp_1277.f9 CaMgO_full_1277.f9
mv CaMgO_full_tmp_1277.gui CaMgO_full_1277.gui
mv CaMgO_full_tmp_1278.f9 CaMgO_full_1278.f9
mv CaMgO_full_tmp_1278.gui CaMgO_full_1278.gui
mv CaMgO_full_tmp_1279.f9 CaMgO_full_1279.f9
mv CaMgO_full_tmp_1279.gui CaMgO_full_1279.gui
mv CaMgO_full_tmp_1280.f9 CaMgO_full_1280.f9
mv CaMgO_full_tmp_1280.gui CaMgO_full_1280.gui
mv CaMgO_full_tmp_1281.f9 CaMgO_full_1281.f9
mv CaMgO_full_tmp_1281.gui CaMgO_full_1281.gui
mv CaMgO_full_tmp_1282.f9 CaMgO_full_1282.f9
mv CaMgO_full_tmp_1282.gui CaMgO_full_1282.gui
mv CaMgO_full_tmp_1283.f9 CaMgO_full_1283.f9
mv CaMgO_full_tmp_1283.gui CaMgO_full_1283.gui
mv CaMgO_full_tmp_1284.f9 CaMgO_full_1284.f9
mv CaMgO_full_tmp_1284.gui CaMgO_full_1284.gui
mv CaMgO_full_tmp_1285.f9 CaMgO_full_1285.f9
mv CaMgO_full_tmp_1285.gui CaMgO_full_1285.gui
mv CaMgO_full_tmp_1286.f9 CaMgO_full_1286.f9
mv CaMgO_full_tmp_1286.gui CaMgO_full_1286.gui
mv CaMgO_full_tmp_1287.f9 CaMgO_full_1287.f9
mv CaMgO_full_tmp_1287.gui CaMgO_full_1287.gui
mv CaMgO_full_tmp_1288.f9 CaMgO_full_1288.f9
mv CaMgO_full_tmp_1288.gui CaMgO_full_1288.gui
mv CaMgO_full_tmp_1289.f9 CaMgO_full_1289.f9
mv CaMgO_full_tmp_1289.gui CaMgO_full_1289.gui
mv CaMgO_full_tmp_1290.f9 CaMgO_full_1290.f9
mv CaMgO_full_tmp_1290.gui CaMgO_full_1290.gui
mv CaMgO_full_tmp_1291.f9 CaMgO_full_1291.f9
mv CaMgO_full_tmp_1291.gui CaMgO_full_1291.gui
mv CaMgO_full_tmp_1292.f9 CaMgO_full_1292.f9
mv CaMgO_full_tmp_1292.gui CaMgO_full_1292.gui
mv CaMgO_full_tmp_1293.f9 CaMgO_full_1293.f9
mv CaMgO_full_tmp_1293.gui CaMgO_full_1293.gui
mv CaMgO_full_tmp_1294.f9 CaMgO_full_1294.f9
mv CaMgO_full_tmp_1294.gui CaMgO_full_1294.gui
mv CaMgO_full_tmp_1295.f9 CaMgO_full_1295.f9
mv CaMgO_full_tmp_1295.gui CaMgO_full_1295.gui
mv CaMgO_full_tmp_1296.f9 CaMgO_full_1296.f9
mv CaMgO_full_tmp_1296.gui CaMgO_full_1296.gui
mv CaMgO_full_tmp_1297.f9 CaMgO_full_1297.f9
mv CaMgO_full_tmp_1297.gui CaMgO_full_1297.gui
mv CaMgO_full_tmp_1298.f9 CaMgO_full_1298.f9
mv CaMgO_full_tmp_1298.gui CaMgO_full_1298.gui
mv CaMgO_full_tmp_1299.f9 CaMgO_full_1299.f9
mv CaMgO_full_tmp_1299.gui CaMgO_full_1299.gui
