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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1401 CaMgO_coord_1401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1402 CaMgO_coord_1402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1403 CaMgO_coord_1403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1404 CaMgO_coord_1404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1405 CaMgO_coord_1405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1406 CaMgO_coord_1406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1407 CaMgO_coord_1407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1408 CaMgO_coord_1408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1409 CaMgO_coord_1409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1410 CaMgO_coord_1410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1411 CaMgO_coord_1411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1412 CaMgO_coord_1412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1413 CaMgO_coord_1413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1414 CaMgO_coord_1414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1415 CaMgO_coord_1415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1416 CaMgO_coord_1416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1417 CaMgO_coord_1417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1418 CaMgO_coord_1418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1419 CaMgO_coord_1419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1420 CaMgO_coord_1420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1421 CaMgO_coord_1421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1422 CaMgO_coord_1422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1423 CaMgO_coord_1423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1424 CaMgO_coord_1424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1425 CaMgO_coord_1425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1426 CaMgO_coord_1426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1427 CaMgO_coord_1427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1428 CaMgO_coord_1428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1429 CaMgO_coord_1429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1430 CaMgO_coord_1430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1431 CaMgO_coord_1431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1432 CaMgO_coord_1432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1433 CaMgO_coord_1433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1434 CaMgO_coord_1434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1435 CaMgO_coord_1435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1436 CaMgO_coord_1436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1437 CaMgO_coord_1437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1438 CaMgO_coord_1438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1439 CaMgO_coord_1439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1440 CaMgO_coord_1440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1441 CaMgO_coord_1441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1442 CaMgO_coord_1442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1443 CaMgO_coord_1443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1444 CaMgO_coord_1444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1445 CaMgO_coord_1445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1446 CaMgO_coord_1446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1447 CaMgO_coord_1447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1448 CaMgO_coord_1448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1449 CaMgO_coord_1449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1450 CaMgO_coord_1450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1451 CaMgO_coord_1451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1452 CaMgO_coord_1452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1453 CaMgO_coord_1453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1454 CaMgO_coord_1454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1455 CaMgO_coord_1455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1456 CaMgO_coord_1456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1457 CaMgO_coord_1457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1458 CaMgO_coord_1458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1459 CaMgO_coord_1459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1460 CaMgO_coord_1460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1461 CaMgO_coord_1461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1462 CaMgO_coord_1462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1463 CaMgO_coord_1463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1464 CaMgO_coord_1464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1465 CaMgO_coord_1465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1466 CaMgO_coord_1466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1467 CaMgO_coord_1467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1468 CaMgO_coord_1468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1469 CaMgO_coord_1469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1470 CaMgO_coord_1470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1471 CaMgO_coord_1471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1472 CaMgO_coord_1472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1473 CaMgO_coord_1473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1474 CaMgO_coord_1474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1475 CaMgO_coord_1475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1476 CaMgO_coord_1476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1477 CaMgO_coord_1477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1478 CaMgO_coord_1478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1479 CaMgO_coord_1479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1480 CaMgO_coord_1480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1481 CaMgO_coord_1481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1482 CaMgO_coord_1482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1483 CaMgO_coord_1483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1484 CaMgO_coord_1484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1485 CaMgO_coord_1485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1486 CaMgO_coord_1486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1487 CaMgO_coord_1487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1488 CaMgO_coord_1488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1489 CaMgO_coord_1489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1490 CaMgO_coord_1490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1491 CaMgO_coord_1491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1492 CaMgO_coord_1492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1493 CaMgO_coord_1493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1494 CaMgO_coord_1494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1495 CaMgO_coord_1495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1496 CaMgO_coord_1496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1497 CaMgO_coord_1497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1498 CaMgO_coord_1498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1499 CaMgO_coord_1499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1499


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1500 CaMgO_coord_1500
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1500


