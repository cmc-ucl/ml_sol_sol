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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1301 CaMgO_coord_tmp_1301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1302 CaMgO_coord_tmp_1302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1303 CaMgO_coord_tmp_1303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1304 CaMgO_coord_tmp_1304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1305 CaMgO_coord_tmp_1305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1306 CaMgO_coord_tmp_1306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1307 CaMgO_coord_tmp_1307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1308 CaMgO_coord_tmp_1308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1309 CaMgO_coord_tmp_1309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1310 CaMgO_coord_tmp_1310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1311 CaMgO_coord_tmp_1311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1312 CaMgO_coord_tmp_1312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1313 CaMgO_coord_tmp_1313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1314 CaMgO_coord_tmp_1314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1315 CaMgO_coord_tmp_1315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1316 CaMgO_coord_tmp_1316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1317 CaMgO_coord_tmp_1317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1318 CaMgO_coord_tmp_1318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1319 CaMgO_coord_tmp_1319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1320 CaMgO_coord_tmp_1320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1321 CaMgO_coord_tmp_1321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1322 CaMgO_coord_tmp_1322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1323 CaMgO_coord_tmp_1323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1324 CaMgO_coord_tmp_1324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1325 CaMgO_coord_tmp_1325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1326 CaMgO_coord_tmp_1326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1327 CaMgO_coord_tmp_1327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1328 CaMgO_coord_tmp_1328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1329 CaMgO_coord_tmp_1329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1330 CaMgO_coord_tmp_1330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1331 CaMgO_coord_tmp_1331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1332 CaMgO_coord_tmp_1332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1333 CaMgO_coord_tmp_1333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1334 CaMgO_coord_tmp_1334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1335 CaMgO_coord_tmp_1335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1336 CaMgO_coord_tmp_1336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1337 CaMgO_coord_tmp_1337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1338 CaMgO_coord_tmp_1338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1339 CaMgO_coord_tmp_1339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1340 CaMgO_coord_tmp_1340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1341 CaMgO_coord_tmp_1341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1342 CaMgO_coord_tmp_1342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1343 CaMgO_coord_tmp_1343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1344 CaMgO_coord_tmp_1344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1345 CaMgO_coord_tmp_1345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1346 CaMgO_coord_tmp_1346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1347 CaMgO_coord_tmp_1347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1348 CaMgO_coord_tmp_1348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1349 CaMgO_coord_tmp_1349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1350 CaMgO_coord_tmp_1350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1351 CaMgO_coord_tmp_1351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1352 CaMgO_coord_tmp_1352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1353 CaMgO_coord_tmp_1353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1354 CaMgO_coord_tmp_1354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1355 CaMgO_coord_tmp_1355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1356 CaMgO_coord_tmp_1356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1357 CaMgO_coord_tmp_1357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1358 CaMgO_coord_tmp_1358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1359 CaMgO_coord_tmp_1359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1360 CaMgO_coord_tmp_1360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1361 CaMgO_coord_tmp_1361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1362 CaMgO_coord_tmp_1362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1363 CaMgO_coord_tmp_1363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1364 CaMgO_coord_tmp_1364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1365 CaMgO_coord_tmp_1365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1366 CaMgO_coord_tmp_1366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1367 CaMgO_coord_tmp_1367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1368 CaMgO_coord_tmp_1368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1369 CaMgO_coord_tmp_1369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1370 CaMgO_coord_tmp_1370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1371 CaMgO_coord_tmp_1371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1372 CaMgO_coord_tmp_1372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1373 CaMgO_coord_tmp_1373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1374 CaMgO_coord_tmp_1374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1375 CaMgO_coord_tmp_1375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1376 CaMgO_coord_tmp_1376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1377 CaMgO_coord_tmp_1377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1378 CaMgO_coord_tmp_1378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1379 CaMgO_coord_tmp_1379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1380 CaMgO_coord_tmp_1380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1381 CaMgO_coord_tmp_1381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1382 CaMgO_coord_tmp_1382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1383 CaMgO_coord_tmp_1383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1384 CaMgO_coord_tmp_1384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1385 CaMgO_coord_tmp_1385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1386 CaMgO_coord_tmp_1386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1387 CaMgO_coord_tmp_1387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1388 CaMgO_coord_tmp_1388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1389 CaMgO_coord_tmp_1389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1390 CaMgO_coord_tmp_1390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1391 CaMgO_coord_tmp_1391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1392 CaMgO_coord_tmp_1392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1393 CaMgO_coord_tmp_1393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1394 CaMgO_coord_tmp_1394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1395 CaMgO_coord_tmp_1395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1396 CaMgO_coord_tmp_1396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1397 CaMgO_coord_tmp_1397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1398 CaMgO_coord_tmp_1398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1399 CaMgO_coord_tmp_1399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1399


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1400 CaMgO_coord_tmp_1400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1400


