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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1399


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1400


