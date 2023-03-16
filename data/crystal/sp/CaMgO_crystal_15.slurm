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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1499


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1500
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1500


