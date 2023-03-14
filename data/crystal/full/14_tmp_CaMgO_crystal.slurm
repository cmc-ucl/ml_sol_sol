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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1300 CaMgO_full_tmp_1300 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1301 CaMgO_full_tmp_1301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1302 CaMgO_full_tmp_1302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1303 CaMgO_full_tmp_1303 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1304 CaMgO_full_tmp_1304 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1305 CaMgO_full_tmp_1305 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1306 CaMgO_full_tmp_1306 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1307 CaMgO_full_tmp_1307 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1308 CaMgO_full_tmp_1308 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1309 CaMgO_full_tmp_1309 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1310 CaMgO_full_tmp_1310 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1311 CaMgO_full_tmp_1311 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1312 CaMgO_full_tmp_1312 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1313 CaMgO_full_tmp_1313 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1314 CaMgO_full_tmp_1314 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1315 CaMgO_full_tmp_1315 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1316 CaMgO_full_tmp_1316 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1317 CaMgO_full_tmp_1317 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1318 CaMgO_full_tmp_1318 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1319 CaMgO_full_tmp_1319 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1320 CaMgO_full_tmp_1320 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1321 CaMgO_full_tmp_1321 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1322 CaMgO_full_tmp_1322 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1323 CaMgO_full_tmp_1323 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1324 CaMgO_full_tmp_1324 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1325 CaMgO_full_tmp_1325 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1326 CaMgO_full_tmp_1326 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1327 CaMgO_full_tmp_1327 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1328 CaMgO_full_tmp_1328 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1329 CaMgO_full_tmp_1329 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1330 CaMgO_full_tmp_1330 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1331 CaMgO_full_tmp_1331 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1332 CaMgO_full_tmp_1332 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1333 CaMgO_full_tmp_1333 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1334 CaMgO_full_tmp_1334 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1335 CaMgO_full_tmp_1335 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1336 CaMgO_full_tmp_1336 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1337 CaMgO_full_tmp_1337 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1338 CaMgO_full_tmp_1338 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1339 CaMgO_full_tmp_1339 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1340 CaMgO_full_tmp_1340 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1341 CaMgO_full_tmp_1341 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1342 CaMgO_full_tmp_1342 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1343 CaMgO_full_tmp_1343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1344 CaMgO_full_tmp_1344 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1345 CaMgO_full_tmp_1345 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1346 CaMgO_full_tmp_1346 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1347 CaMgO_full_tmp_1347 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1348 CaMgO_full_tmp_1348 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1349 CaMgO_full_tmp_1349 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1350 CaMgO_full_tmp_1350 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1351 CaMgO_full_tmp_1351 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1352 CaMgO_full_tmp_1352 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1353 CaMgO_full_tmp_1353 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1354 CaMgO_full_tmp_1354 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1355 CaMgO_full_tmp_1355 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1356 CaMgO_full_tmp_1356 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1357 CaMgO_full_tmp_1357 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1358 CaMgO_full_tmp_1358 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1359 CaMgO_full_tmp_1359 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1360 CaMgO_full_tmp_1360 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1361 CaMgO_full_tmp_1361 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1362 CaMgO_full_tmp_1362 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1363 CaMgO_full_tmp_1363 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1364 CaMgO_full_tmp_1364 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1365 CaMgO_full_tmp_1365 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1366 CaMgO_full_tmp_1366 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1367 CaMgO_full_tmp_1367 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1368 CaMgO_full_tmp_1368 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1369 CaMgO_full_tmp_1369 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1370 CaMgO_full_tmp_1370 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1371 CaMgO_full_tmp_1371 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1372 CaMgO_full_tmp_1372 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1373 CaMgO_full_tmp_1373 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1374 CaMgO_full_tmp_1374 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1375 CaMgO_full_tmp_1375 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1376 CaMgO_full_tmp_1376 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1377 CaMgO_full_tmp_1377 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1378 CaMgO_full_tmp_1378 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1379 CaMgO_full_tmp_1379 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1380 CaMgO_full_tmp_1380 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1381 CaMgO_full_tmp_1381 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1382 CaMgO_full_tmp_1382 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1383 CaMgO_full_tmp_1383 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1384 CaMgO_full_tmp_1384 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1385 CaMgO_full_tmp_1385 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1386 CaMgO_full_tmp_1386 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1387 CaMgO_full_tmp_1387 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1388 CaMgO_full_tmp_1388 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1389 CaMgO_full_tmp_1389 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1390 CaMgO_full_tmp_1390 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1391 CaMgO_full_tmp_1391 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1392 CaMgO_full_tmp_1392 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1393 CaMgO_full_tmp_1393 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1394 CaMgO_full_tmp_1394 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1395 CaMgO_full_tmp_1395 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1396 CaMgO_full_tmp_1396 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1397 CaMgO_full_tmp_1397 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1398 CaMgO_full_tmp_1398 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1399 CaMgO_full_tmp_1399 &

wait
mv CaMgO_full_tmp_1300.f9 CaMgO_full_1300.f9
mv CaMgO_full_tmp_1300.gui CaMgO_full_1300.gui
mv CaMgO_full_tmp_1301.f9 CaMgO_full_1301.f9
mv CaMgO_full_tmp_1301.gui CaMgO_full_1301.gui
mv CaMgO_full_tmp_1302.f9 CaMgO_full_1302.f9
mv CaMgO_full_tmp_1302.gui CaMgO_full_1302.gui
mv CaMgO_full_tmp_1303.f9 CaMgO_full_1303.f9
mv CaMgO_full_tmp_1303.gui CaMgO_full_1303.gui
mv CaMgO_full_tmp_1304.f9 CaMgO_full_1304.f9
mv CaMgO_full_tmp_1304.gui CaMgO_full_1304.gui
mv CaMgO_full_tmp_1305.f9 CaMgO_full_1305.f9
mv CaMgO_full_tmp_1305.gui CaMgO_full_1305.gui
mv CaMgO_full_tmp_1306.f9 CaMgO_full_1306.f9
mv CaMgO_full_tmp_1306.gui CaMgO_full_1306.gui
mv CaMgO_full_tmp_1307.f9 CaMgO_full_1307.f9
mv CaMgO_full_tmp_1307.gui CaMgO_full_1307.gui
mv CaMgO_full_tmp_1308.f9 CaMgO_full_1308.f9
mv CaMgO_full_tmp_1308.gui CaMgO_full_1308.gui
mv CaMgO_full_tmp_1309.f9 CaMgO_full_1309.f9
mv CaMgO_full_tmp_1309.gui CaMgO_full_1309.gui
mv CaMgO_full_tmp_1310.f9 CaMgO_full_1310.f9
mv CaMgO_full_tmp_1310.gui CaMgO_full_1310.gui
mv CaMgO_full_tmp_1311.f9 CaMgO_full_1311.f9
mv CaMgO_full_tmp_1311.gui CaMgO_full_1311.gui
mv CaMgO_full_tmp_1312.f9 CaMgO_full_1312.f9
mv CaMgO_full_tmp_1312.gui CaMgO_full_1312.gui
mv CaMgO_full_tmp_1313.f9 CaMgO_full_1313.f9
mv CaMgO_full_tmp_1313.gui CaMgO_full_1313.gui
mv CaMgO_full_tmp_1314.f9 CaMgO_full_1314.f9
mv CaMgO_full_tmp_1314.gui CaMgO_full_1314.gui
mv CaMgO_full_tmp_1315.f9 CaMgO_full_1315.f9
mv CaMgO_full_tmp_1315.gui CaMgO_full_1315.gui
mv CaMgO_full_tmp_1316.f9 CaMgO_full_1316.f9
mv CaMgO_full_tmp_1316.gui CaMgO_full_1316.gui
mv CaMgO_full_tmp_1317.f9 CaMgO_full_1317.f9
mv CaMgO_full_tmp_1317.gui CaMgO_full_1317.gui
mv CaMgO_full_tmp_1318.f9 CaMgO_full_1318.f9
mv CaMgO_full_tmp_1318.gui CaMgO_full_1318.gui
mv CaMgO_full_tmp_1319.f9 CaMgO_full_1319.f9
mv CaMgO_full_tmp_1319.gui CaMgO_full_1319.gui
mv CaMgO_full_tmp_1320.f9 CaMgO_full_1320.f9
mv CaMgO_full_tmp_1320.gui CaMgO_full_1320.gui
mv CaMgO_full_tmp_1321.f9 CaMgO_full_1321.f9
mv CaMgO_full_tmp_1321.gui CaMgO_full_1321.gui
mv CaMgO_full_tmp_1322.f9 CaMgO_full_1322.f9
mv CaMgO_full_tmp_1322.gui CaMgO_full_1322.gui
mv CaMgO_full_tmp_1323.f9 CaMgO_full_1323.f9
mv CaMgO_full_tmp_1323.gui CaMgO_full_1323.gui
mv CaMgO_full_tmp_1324.f9 CaMgO_full_1324.f9
mv CaMgO_full_tmp_1324.gui CaMgO_full_1324.gui
mv CaMgO_full_tmp_1325.f9 CaMgO_full_1325.f9
mv CaMgO_full_tmp_1325.gui CaMgO_full_1325.gui
mv CaMgO_full_tmp_1326.f9 CaMgO_full_1326.f9
mv CaMgO_full_tmp_1326.gui CaMgO_full_1326.gui
mv CaMgO_full_tmp_1327.f9 CaMgO_full_1327.f9
mv CaMgO_full_tmp_1327.gui CaMgO_full_1327.gui
mv CaMgO_full_tmp_1328.f9 CaMgO_full_1328.f9
mv CaMgO_full_tmp_1328.gui CaMgO_full_1328.gui
mv CaMgO_full_tmp_1329.f9 CaMgO_full_1329.f9
mv CaMgO_full_tmp_1329.gui CaMgO_full_1329.gui
mv CaMgO_full_tmp_1330.f9 CaMgO_full_1330.f9
mv CaMgO_full_tmp_1330.gui CaMgO_full_1330.gui
mv CaMgO_full_tmp_1331.f9 CaMgO_full_1331.f9
mv CaMgO_full_tmp_1331.gui CaMgO_full_1331.gui
mv CaMgO_full_tmp_1332.f9 CaMgO_full_1332.f9
mv CaMgO_full_tmp_1332.gui CaMgO_full_1332.gui
mv CaMgO_full_tmp_1333.f9 CaMgO_full_1333.f9
mv CaMgO_full_tmp_1333.gui CaMgO_full_1333.gui
mv CaMgO_full_tmp_1334.f9 CaMgO_full_1334.f9
mv CaMgO_full_tmp_1334.gui CaMgO_full_1334.gui
mv CaMgO_full_tmp_1335.f9 CaMgO_full_1335.f9
mv CaMgO_full_tmp_1335.gui CaMgO_full_1335.gui
mv CaMgO_full_tmp_1336.f9 CaMgO_full_1336.f9
mv CaMgO_full_tmp_1336.gui CaMgO_full_1336.gui
mv CaMgO_full_tmp_1337.f9 CaMgO_full_1337.f9
mv CaMgO_full_tmp_1337.gui CaMgO_full_1337.gui
mv CaMgO_full_tmp_1338.f9 CaMgO_full_1338.f9
mv CaMgO_full_tmp_1338.gui CaMgO_full_1338.gui
mv CaMgO_full_tmp_1339.f9 CaMgO_full_1339.f9
mv CaMgO_full_tmp_1339.gui CaMgO_full_1339.gui
mv CaMgO_full_tmp_1340.f9 CaMgO_full_1340.f9
mv CaMgO_full_tmp_1340.gui CaMgO_full_1340.gui
mv CaMgO_full_tmp_1341.f9 CaMgO_full_1341.f9
mv CaMgO_full_tmp_1341.gui CaMgO_full_1341.gui
mv CaMgO_full_tmp_1342.f9 CaMgO_full_1342.f9
mv CaMgO_full_tmp_1342.gui CaMgO_full_1342.gui
mv CaMgO_full_tmp_1343.f9 CaMgO_full_1343.f9
mv CaMgO_full_tmp_1343.gui CaMgO_full_1343.gui
mv CaMgO_full_tmp_1344.f9 CaMgO_full_1344.f9
mv CaMgO_full_tmp_1344.gui CaMgO_full_1344.gui
mv CaMgO_full_tmp_1345.f9 CaMgO_full_1345.f9
mv CaMgO_full_tmp_1345.gui CaMgO_full_1345.gui
mv CaMgO_full_tmp_1346.f9 CaMgO_full_1346.f9
mv CaMgO_full_tmp_1346.gui CaMgO_full_1346.gui
mv CaMgO_full_tmp_1347.f9 CaMgO_full_1347.f9
mv CaMgO_full_tmp_1347.gui CaMgO_full_1347.gui
mv CaMgO_full_tmp_1348.f9 CaMgO_full_1348.f9
mv CaMgO_full_tmp_1348.gui CaMgO_full_1348.gui
mv CaMgO_full_tmp_1349.f9 CaMgO_full_1349.f9
mv CaMgO_full_tmp_1349.gui CaMgO_full_1349.gui
mv CaMgO_full_tmp_1350.f9 CaMgO_full_1350.f9
mv CaMgO_full_tmp_1350.gui CaMgO_full_1350.gui
mv CaMgO_full_tmp_1351.f9 CaMgO_full_1351.f9
mv CaMgO_full_tmp_1351.gui CaMgO_full_1351.gui
mv CaMgO_full_tmp_1352.f9 CaMgO_full_1352.f9
mv CaMgO_full_tmp_1352.gui CaMgO_full_1352.gui
mv CaMgO_full_tmp_1353.f9 CaMgO_full_1353.f9
mv CaMgO_full_tmp_1353.gui CaMgO_full_1353.gui
mv CaMgO_full_tmp_1354.f9 CaMgO_full_1354.f9
mv CaMgO_full_tmp_1354.gui CaMgO_full_1354.gui
mv CaMgO_full_tmp_1355.f9 CaMgO_full_1355.f9
mv CaMgO_full_tmp_1355.gui CaMgO_full_1355.gui
mv CaMgO_full_tmp_1356.f9 CaMgO_full_1356.f9
mv CaMgO_full_tmp_1356.gui CaMgO_full_1356.gui
mv CaMgO_full_tmp_1357.f9 CaMgO_full_1357.f9
mv CaMgO_full_tmp_1357.gui CaMgO_full_1357.gui
mv CaMgO_full_tmp_1358.f9 CaMgO_full_1358.f9
mv CaMgO_full_tmp_1358.gui CaMgO_full_1358.gui
mv CaMgO_full_tmp_1359.f9 CaMgO_full_1359.f9
mv CaMgO_full_tmp_1359.gui CaMgO_full_1359.gui
mv CaMgO_full_tmp_1360.f9 CaMgO_full_1360.f9
mv CaMgO_full_tmp_1360.gui CaMgO_full_1360.gui
mv CaMgO_full_tmp_1361.f9 CaMgO_full_1361.f9
mv CaMgO_full_tmp_1361.gui CaMgO_full_1361.gui
mv CaMgO_full_tmp_1362.f9 CaMgO_full_1362.f9
mv CaMgO_full_tmp_1362.gui CaMgO_full_1362.gui
mv CaMgO_full_tmp_1363.f9 CaMgO_full_1363.f9
mv CaMgO_full_tmp_1363.gui CaMgO_full_1363.gui
mv CaMgO_full_tmp_1364.f9 CaMgO_full_1364.f9
mv CaMgO_full_tmp_1364.gui CaMgO_full_1364.gui
mv CaMgO_full_tmp_1365.f9 CaMgO_full_1365.f9
mv CaMgO_full_tmp_1365.gui CaMgO_full_1365.gui
mv CaMgO_full_tmp_1366.f9 CaMgO_full_1366.f9
mv CaMgO_full_tmp_1366.gui CaMgO_full_1366.gui
mv CaMgO_full_tmp_1367.f9 CaMgO_full_1367.f9
mv CaMgO_full_tmp_1367.gui CaMgO_full_1367.gui
mv CaMgO_full_tmp_1368.f9 CaMgO_full_1368.f9
mv CaMgO_full_tmp_1368.gui CaMgO_full_1368.gui
mv CaMgO_full_tmp_1369.f9 CaMgO_full_1369.f9
mv CaMgO_full_tmp_1369.gui CaMgO_full_1369.gui
mv CaMgO_full_tmp_1370.f9 CaMgO_full_1370.f9
mv CaMgO_full_tmp_1370.gui CaMgO_full_1370.gui
mv CaMgO_full_tmp_1371.f9 CaMgO_full_1371.f9
mv CaMgO_full_tmp_1371.gui CaMgO_full_1371.gui
mv CaMgO_full_tmp_1372.f9 CaMgO_full_1372.f9
mv CaMgO_full_tmp_1372.gui CaMgO_full_1372.gui
mv CaMgO_full_tmp_1373.f9 CaMgO_full_1373.f9
mv CaMgO_full_tmp_1373.gui CaMgO_full_1373.gui
mv CaMgO_full_tmp_1374.f9 CaMgO_full_1374.f9
mv CaMgO_full_tmp_1374.gui CaMgO_full_1374.gui
mv CaMgO_full_tmp_1375.f9 CaMgO_full_1375.f9
mv CaMgO_full_tmp_1375.gui CaMgO_full_1375.gui
mv CaMgO_full_tmp_1376.f9 CaMgO_full_1376.f9
mv CaMgO_full_tmp_1376.gui CaMgO_full_1376.gui
mv CaMgO_full_tmp_1377.f9 CaMgO_full_1377.f9
mv CaMgO_full_tmp_1377.gui CaMgO_full_1377.gui
mv CaMgO_full_tmp_1378.f9 CaMgO_full_1378.f9
mv CaMgO_full_tmp_1378.gui CaMgO_full_1378.gui
mv CaMgO_full_tmp_1379.f9 CaMgO_full_1379.f9
mv CaMgO_full_tmp_1379.gui CaMgO_full_1379.gui
mv CaMgO_full_tmp_1380.f9 CaMgO_full_1380.f9
mv CaMgO_full_tmp_1380.gui CaMgO_full_1380.gui
mv CaMgO_full_tmp_1381.f9 CaMgO_full_1381.f9
mv CaMgO_full_tmp_1381.gui CaMgO_full_1381.gui
mv CaMgO_full_tmp_1382.f9 CaMgO_full_1382.f9
mv CaMgO_full_tmp_1382.gui CaMgO_full_1382.gui
mv CaMgO_full_tmp_1383.f9 CaMgO_full_1383.f9
mv CaMgO_full_tmp_1383.gui CaMgO_full_1383.gui
mv CaMgO_full_tmp_1384.f9 CaMgO_full_1384.f9
mv CaMgO_full_tmp_1384.gui CaMgO_full_1384.gui
mv CaMgO_full_tmp_1385.f9 CaMgO_full_1385.f9
mv CaMgO_full_tmp_1385.gui CaMgO_full_1385.gui
mv CaMgO_full_tmp_1386.f9 CaMgO_full_1386.f9
mv CaMgO_full_tmp_1386.gui CaMgO_full_1386.gui
mv CaMgO_full_tmp_1387.f9 CaMgO_full_1387.f9
mv CaMgO_full_tmp_1387.gui CaMgO_full_1387.gui
mv CaMgO_full_tmp_1388.f9 CaMgO_full_1388.f9
mv CaMgO_full_tmp_1388.gui CaMgO_full_1388.gui
mv CaMgO_full_tmp_1389.f9 CaMgO_full_1389.f9
mv CaMgO_full_tmp_1389.gui CaMgO_full_1389.gui
mv CaMgO_full_tmp_1390.f9 CaMgO_full_1390.f9
mv CaMgO_full_tmp_1390.gui CaMgO_full_1390.gui
mv CaMgO_full_tmp_1391.f9 CaMgO_full_1391.f9
mv CaMgO_full_tmp_1391.gui CaMgO_full_1391.gui
mv CaMgO_full_tmp_1392.f9 CaMgO_full_1392.f9
mv CaMgO_full_tmp_1392.gui CaMgO_full_1392.gui
mv CaMgO_full_tmp_1393.f9 CaMgO_full_1393.f9
mv CaMgO_full_tmp_1393.gui CaMgO_full_1393.gui
mv CaMgO_full_tmp_1394.f9 CaMgO_full_1394.f9
mv CaMgO_full_tmp_1394.gui CaMgO_full_1394.gui
mv CaMgO_full_tmp_1395.f9 CaMgO_full_1395.f9
mv CaMgO_full_tmp_1395.gui CaMgO_full_1395.gui
mv CaMgO_full_tmp_1396.f9 CaMgO_full_1396.f9
mv CaMgO_full_tmp_1396.gui CaMgO_full_1396.gui
mv CaMgO_full_tmp_1397.f9 CaMgO_full_1397.f9
mv CaMgO_full_tmp_1397.gui CaMgO_full_1397.gui
mv CaMgO_full_tmp_1398.f9 CaMgO_full_1398.f9
mv CaMgO_full_tmp_1398.gui CaMgO_full_1398.gui
mv CaMgO_full_tmp_1399.f9 CaMgO_full_1399.f9
mv CaMgO_full_tmp_1399.gui CaMgO_full_1399.gui
