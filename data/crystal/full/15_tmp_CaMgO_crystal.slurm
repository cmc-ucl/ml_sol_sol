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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1400 CaMgO_full_tmp_1400 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1401 CaMgO_full_tmp_1401 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1402 CaMgO_full_tmp_1402 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1403 CaMgO_full_tmp_1403 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1404 CaMgO_full_tmp_1404 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1405 CaMgO_full_tmp_1405 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1406 CaMgO_full_tmp_1406 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1407 CaMgO_full_tmp_1407 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1408 CaMgO_full_tmp_1408 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1409 CaMgO_full_tmp_1409 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1410 CaMgO_full_tmp_1410 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1411 CaMgO_full_tmp_1411 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1412 CaMgO_full_tmp_1412 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1413 CaMgO_full_tmp_1413 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1414 CaMgO_full_tmp_1414 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1415 CaMgO_full_tmp_1415 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1416 CaMgO_full_tmp_1416 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1417 CaMgO_full_tmp_1417 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1418 CaMgO_full_tmp_1418 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1419 CaMgO_full_tmp_1419 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1420 CaMgO_full_tmp_1420 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1421 CaMgO_full_tmp_1421 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1422 CaMgO_full_tmp_1422 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1423 CaMgO_full_tmp_1423 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1424 CaMgO_full_tmp_1424 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1425 CaMgO_full_tmp_1425 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1426 CaMgO_full_tmp_1426 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1427 CaMgO_full_tmp_1427 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1428 CaMgO_full_tmp_1428 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1429 CaMgO_full_tmp_1429 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1430 CaMgO_full_tmp_1430 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1431 CaMgO_full_tmp_1431 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1432 CaMgO_full_tmp_1432 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1433 CaMgO_full_tmp_1433 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1434 CaMgO_full_tmp_1434 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1435 CaMgO_full_tmp_1435 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1436 CaMgO_full_tmp_1436 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1437 CaMgO_full_tmp_1437 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1438 CaMgO_full_tmp_1438 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1439 CaMgO_full_tmp_1439 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1440 CaMgO_full_tmp_1440 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1441 CaMgO_full_tmp_1441 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1442 CaMgO_full_tmp_1442 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1443 CaMgO_full_tmp_1443 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1444 CaMgO_full_tmp_1444 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1445 CaMgO_full_tmp_1445 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1446 CaMgO_full_tmp_1446 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1447 CaMgO_full_tmp_1447 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1448 CaMgO_full_tmp_1448 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1449 CaMgO_full_tmp_1449 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1450 CaMgO_full_tmp_1450 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1451 CaMgO_full_tmp_1451 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1452 CaMgO_full_tmp_1452 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1453 CaMgO_full_tmp_1453 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1454 CaMgO_full_tmp_1454 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1455 CaMgO_full_tmp_1455 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1456 CaMgO_full_tmp_1456 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1457 CaMgO_full_tmp_1457 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1458 CaMgO_full_tmp_1458 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1459 CaMgO_full_tmp_1459 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1460 CaMgO_full_tmp_1460 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1461 CaMgO_full_tmp_1461 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1462 CaMgO_full_tmp_1462 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1463 CaMgO_full_tmp_1463 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1464 CaMgO_full_tmp_1464 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1465 CaMgO_full_tmp_1465 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1466 CaMgO_full_tmp_1466 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1467 CaMgO_full_tmp_1467 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1468 CaMgO_full_tmp_1468 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1469 CaMgO_full_tmp_1469 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1470 CaMgO_full_tmp_1470 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1471 CaMgO_full_tmp_1471 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1472 CaMgO_full_tmp_1472 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1473 CaMgO_full_tmp_1473 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1474 CaMgO_full_tmp_1474 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1475 CaMgO_full_tmp_1475 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1476 CaMgO_full_tmp_1476 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1477 CaMgO_full_tmp_1477 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1478 CaMgO_full_tmp_1478 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1479 CaMgO_full_tmp_1479 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1480 CaMgO_full_tmp_1480 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1481 CaMgO_full_tmp_1481 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1482 CaMgO_full_tmp_1482 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1483 CaMgO_full_tmp_1483 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1484 CaMgO_full_tmp_1484 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1485 CaMgO_full_tmp_1485 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1486 CaMgO_full_tmp_1486 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1487 CaMgO_full_tmp_1487 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1488 CaMgO_full_tmp_1488 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1489 CaMgO_full_tmp_1489 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1490 CaMgO_full_tmp_1490 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1491 CaMgO_full_tmp_1491 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1492 CaMgO_full_tmp_1492 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1493 CaMgO_full_tmp_1493 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1494 CaMgO_full_tmp_1494 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1495 CaMgO_full_tmp_1495 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1496 CaMgO_full_tmp_1496 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1497 CaMgO_full_tmp_1497 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1498 CaMgO_full_tmp_1498 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1499 CaMgO_full_tmp_1499 &

wait
mv CaMgO_full_tmp_1400.f9 CaMgO_full_1400.f9
mv CaMgO_full_tmp_1400.gui CaMgO_full_1400.gui
mv CaMgO_full_tmp_1401.f9 CaMgO_full_1401.f9
mv CaMgO_full_tmp_1401.gui CaMgO_full_1401.gui
mv CaMgO_full_tmp_1402.f9 CaMgO_full_1402.f9
mv CaMgO_full_tmp_1402.gui CaMgO_full_1402.gui
mv CaMgO_full_tmp_1403.f9 CaMgO_full_1403.f9
mv CaMgO_full_tmp_1403.gui CaMgO_full_1403.gui
mv CaMgO_full_tmp_1404.f9 CaMgO_full_1404.f9
mv CaMgO_full_tmp_1404.gui CaMgO_full_1404.gui
mv CaMgO_full_tmp_1405.f9 CaMgO_full_1405.f9
mv CaMgO_full_tmp_1405.gui CaMgO_full_1405.gui
mv CaMgO_full_tmp_1406.f9 CaMgO_full_1406.f9
mv CaMgO_full_tmp_1406.gui CaMgO_full_1406.gui
mv CaMgO_full_tmp_1407.f9 CaMgO_full_1407.f9
mv CaMgO_full_tmp_1407.gui CaMgO_full_1407.gui
mv CaMgO_full_tmp_1408.f9 CaMgO_full_1408.f9
mv CaMgO_full_tmp_1408.gui CaMgO_full_1408.gui
mv CaMgO_full_tmp_1409.f9 CaMgO_full_1409.f9
mv CaMgO_full_tmp_1409.gui CaMgO_full_1409.gui
mv CaMgO_full_tmp_1410.f9 CaMgO_full_1410.f9
mv CaMgO_full_tmp_1410.gui CaMgO_full_1410.gui
mv CaMgO_full_tmp_1411.f9 CaMgO_full_1411.f9
mv CaMgO_full_tmp_1411.gui CaMgO_full_1411.gui
mv CaMgO_full_tmp_1412.f9 CaMgO_full_1412.f9
mv CaMgO_full_tmp_1412.gui CaMgO_full_1412.gui
mv CaMgO_full_tmp_1413.f9 CaMgO_full_1413.f9
mv CaMgO_full_tmp_1413.gui CaMgO_full_1413.gui
mv CaMgO_full_tmp_1414.f9 CaMgO_full_1414.f9
mv CaMgO_full_tmp_1414.gui CaMgO_full_1414.gui
mv CaMgO_full_tmp_1415.f9 CaMgO_full_1415.f9
mv CaMgO_full_tmp_1415.gui CaMgO_full_1415.gui
mv CaMgO_full_tmp_1416.f9 CaMgO_full_1416.f9
mv CaMgO_full_tmp_1416.gui CaMgO_full_1416.gui
mv CaMgO_full_tmp_1417.f9 CaMgO_full_1417.f9
mv CaMgO_full_tmp_1417.gui CaMgO_full_1417.gui
mv CaMgO_full_tmp_1418.f9 CaMgO_full_1418.f9
mv CaMgO_full_tmp_1418.gui CaMgO_full_1418.gui
mv CaMgO_full_tmp_1419.f9 CaMgO_full_1419.f9
mv CaMgO_full_tmp_1419.gui CaMgO_full_1419.gui
mv CaMgO_full_tmp_1420.f9 CaMgO_full_1420.f9
mv CaMgO_full_tmp_1420.gui CaMgO_full_1420.gui
mv CaMgO_full_tmp_1421.f9 CaMgO_full_1421.f9
mv CaMgO_full_tmp_1421.gui CaMgO_full_1421.gui
mv CaMgO_full_tmp_1422.f9 CaMgO_full_1422.f9
mv CaMgO_full_tmp_1422.gui CaMgO_full_1422.gui
mv CaMgO_full_tmp_1423.f9 CaMgO_full_1423.f9
mv CaMgO_full_tmp_1423.gui CaMgO_full_1423.gui
mv CaMgO_full_tmp_1424.f9 CaMgO_full_1424.f9
mv CaMgO_full_tmp_1424.gui CaMgO_full_1424.gui
mv CaMgO_full_tmp_1425.f9 CaMgO_full_1425.f9
mv CaMgO_full_tmp_1425.gui CaMgO_full_1425.gui
mv CaMgO_full_tmp_1426.f9 CaMgO_full_1426.f9
mv CaMgO_full_tmp_1426.gui CaMgO_full_1426.gui
mv CaMgO_full_tmp_1427.f9 CaMgO_full_1427.f9
mv CaMgO_full_tmp_1427.gui CaMgO_full_1427.gui
mv CaMgO_full_tmp_1428.f9 CaMgO_full_1428.f9
mv CaMgO_full_tmp_1428.gui CaMgO_full_1428.gui
mv CaMgO_full_tmp_1429.f9 CaMgO_full_1429.f9
mv CaMgO_full_tmp_1429.gui CaMgO_full_1429.gui
mv CaMgO_full_tmp_1430.f9 CaMgO_full_1430.f9
mv CaMgO_full_tmp_1430.gui CaMgO_full_1430.gui
mv CaMgO_full_tmp_1431.f9 CaMgO_full_1431.f9
mv CaMgO_full_tmp_1431.gui CaMgO_full_1431.gui
mv CaMgO_full_tmp_1432.f9 CaMgO_full_1432.f9
mv CaMgO_full_tmp_1432.gui CaMgO_full_1432.gui
mv CaMgO_full_tmp_1433.f9 CaMgO_full_1433.f9
mv CaMgO_full_tmp_1433.gui CaMgO_full_1433.gui
mv CaMgO_full_tmp_1434.f9 CaMgO_full_1434.f9
mv CaMgO_full_tmp_1434.gui CaMgO_full_1434.gui
mv CaMgO_full_tmp_1435.f9 CaMgO_full_1435.f9
mv CaMgO_full_tmp_1435.gui CaMgO_full_1435.gui
mv CaMgO_full_tmp_1436.f9 CaMgO_full_1436.f9
mv CaMgO_full_tmp_1436.gui CaMgO_full_1436.gui
mv CaMgO_full_tmp_1437.f9 CaMgO_full_1437.f9
mv CaMgO_full_tmp_1437.gui CaMgO_full_1437.gui
mv CaMgO_full_tmp_1438.f9 CaMgO_full_1438.f9
mv CaMgO_full_tmp_1438.gui CaMgO_full_1438.gui
mv CaMgO_full_tmp_1439.f9 CaMgO_full_1439.f9
mv CaMgO_full_tmp_1439.gui CaMgO_full_1439.gui
mv CaMgO_full_tmp_1440.f9 CaMgO_full_1440.f9
mv CaMgO_full_tmp_1440.gui CaMgO_full_1440.gui
mv CaMgO_full_tmp_1441.f9 CaMgO_full_1441.f9
mv CaMgO_full_tmp_1441.gui CaMgO_full_1441.gui
mv CaMgO_full_tmp_1442.f9 CaMgO_full_1442.f9
mv CaMgO_full_tmp_1442.gui CaMgO_full_1442.gui
mv CaMgO_full_tmp_1443.f9 CaMgO_full_1443.f9
mv CaMgO_full_tmp_1443.gui CaMgO_full_1443.gui
mv CaMgO_full_tmp_1444.f9 CaMgO_full_1444.f9
mv CaMgO_full_tmp_1444.gui CaMgO_full_1444.gui
mv CaMgO_full_tmp_1445.f9 CaMgO_full_1445.f9
mv CaMgO_full_tmp_1445.gui CaMgO_full_1445.gui
mv CaMgO_full_tmp_1446.f9 CaMgO_full_1446.f9
mv CaMgO_full_tmp_1446.gui CaMgO_full_1446.gui
mv CaMgO_full_tmp_1447.f9 CaMgO_full_1447.f9
mv CaMgO_full_tmp_1447.gui CaMgO_full_1447.gui
mv CaMgO_full_tmp_1448.f9 CaMgO_full_1448.f9
mv CaMgO_full_tmp_1448.gui CaMgO_full_1448.gui
mv CaMgO_full_tmp_1449.f9 CaMgO_full_1449.f9
mv CaMgO_full_tmp_1449.gui CaMgO_full_1449.gui
mv CaMgO_full_tmp_1450.f9 CaMgO_full_1450.f9
mv CaMgO_full_tmp_1450.gui CaMgO_full_1450.gui
mv CaMgO_full_tmp_1451.f9 CaMgO_full_1451.f9
mv CaMgO_full_tmp_1451.gui CaMgO_full_1451.gui
mv CaMgO_full_tmp_1452.f9 CaMgO_full_1452.f9
mv CaMgO_full_tmp_1452.gui CaMgO_full_1452.gui
mv CaMgO_full_tmp_1453.f9 CaMgO_full_1453.f9
mv CaMgO_full_tmp_1453.gui CaMgO_full_1453.gui
mv CaMgO_full_tmp_1454.f9 CaMgO_full_1454.f9
mv CaMgO_full_tmp_1454.gui CaMgO_full_1454.gui
mv CaMgO_full_tmp_1455.f9 CaMgO_full_1455.f9
mv CaMgO_full_tmp_1455.gui CaMgO_full_1455.gui
mv CaMgO_full_tmp_1456.f9 CaMgO_full_1456.f9
mv CaMgO_full_tmp_1456.gui CaMgO_full_1456.gui
mv CaMgO_full_tmp_1457.f9 CaMgO_full_1457.f9
mv CaMgO_full_tmp_1457.gui CaMgO_full_1457.gui
mv CaMgO_full_tmp_1458.f9 CaMgO_full_1458.f9
mv CaMgO_full_tmp_1458.gui CaMgO_full_1458.gui
mv CaMgO_full_tmp_1459.f9 CaMgO_full_1459.f9
mv CaMgO_full_tmp_1459.gui CaMgO_full_1459.gui
mv CaMgO_full_tmp_1460.f9 CaMgO_full_1460.f9
mv CaMgO_full_tmp_1460.gui CaMgO_full_1460.gui
mv CaMgO_full_tmp_1461.f9 CaMgO_full_1461.f9
mv CaMgO_full_tmp_1461.gui CaMgO_full_1461.gui
mv CaMgO_full_tmp_1462.f9 CaMgO_full_1462.f9
mv CaMgO_full_tmp_1462.gui CaMgO_full_1462.gui
mv CaMgO_full_tmp_1463.f9 CaMgO_full_1463.f9
mv CaMgO_full_tmp_1463.gui CaMgO_full_1463.gui
mv CaMgO_full_tmp_1464.f9 CaMgO_full_1464.f9
mv CaMgO_full_tmp_1464.gui CaMgO_full_1464.gui
mv CaMgO_full_tmp_1465.f9 CaMgO_full_1465.f9
mv CaMgO_full_tmp_1465.gui CaMgO_full_1465.gui
mv CaMgO_full_tmp_1466.f9 CaMgO_full_1466.f9
mv CaMgO_full_tmp_1466.gui CaMgO_full_1466.gui
mv CaMgO_full_tmp_1467.f9 CaMgO_full_1467.f9
mv CaMgO_full_tmp_1467.gui CaMgO_full_1467.gui
mv CaMgO_full_tmp_1468.f9 CaMgO_full_1468.f9
mv CaMgO_full_tmp_1468.gui CaMgO_full_1468.gui
mv CaMgO_full_tmp_1469.f9 CaMgO_full_1469.f9
mv CaMgO_full_tmp_1469.gui CaMgO_full_1469.gui
mv CaMgO_full_tmp_1470.f9 CaMgO_full_1470.f9
mv CaMgO_full_tmp_1470.gui CaMgO_full_1470.gui
mv CaMgO_full_tmp_1471.f9 CaMgO_full_1471.f9
mv CaMgO_full_tmp_1471.gui CaMgO_full_1471.gui
mv CaMgO_full_tmp_1472.f9 CaMgO_full_1472.f9
mv CaMgO_full_tmp_1472.gui CaMgO_full_1472.gui
mv CaMgO_full_tmp_1473.f9 CaMgO_full_1473.f9
mv CaMgO_full_tmp_1473.gui CaMgO_full_1473.gui
mv CaMgO_full_tmp_1474.f9 CaMgO_full_1474.f9
mv CaMgO_full_tmp_1474.gui CaMgO_full_1474.gui
mv CaMgO_full_tmp_1475.f9 CaMgO_full_1475.f9
mv CaMgO_full_tmp_1475.gui CaMgO_full_1475.gui
mv CaMgO_full_tmp_1476.f9 CaMgO_full_1476.f9
mv CaMgO_full_tmp_1476.gui CaMgO_full_1476.gui
mv CaMgO_full_tmp_1477.f9 CaMgO_full_1477.f9
mv CaMgO_full_tmp_1477.gui CaMgO_full_1477.gui
mv CaMgO_full_tmp_1478.f9 CaMgO_full_1478.f9
mv CaMgO_full_tmp_1478.gui CaMgO_full_1478.gui
mv CaMgO_full_tmp_1479.f9 CaMgO_full_1479.f9
mv CaMgO_full_tmp_1479.gui CaMgO_full_1479.gui
mv CaMgO_full_tmp_1480.f9 CaMgO_full_1480.f9
mv CaMgO_full_tmp_1480.gui CaMgO_full_1480.gui
mv CaMgO_full_tmp_1481.f9 CaMgO_full_1481.f9
mv CaMgO_full_tmp_1481.gui CaMgO_full_1481.gui
mv CaMgO_full_tmp_1482.f9 CaMgO_full_1482.f9
mv CaMgO_full_tmp_1482.gui CaMgO_full_1482.gui
mv CaMgO_full_tmp_1483.f9 CaMgO_full_1483.f9
mv CaMgO_full_tmp_1483.gui CaMgO_full_1483.gui
mv CaMgO_full_tmp_1484.f9 CaMgO_full_1484.f9
mv CaMgO_full_tmp_1484.gui CaMgO_full_1484.gui
mv CaMgO_full_tmp_1485.f9 CaMgO_full_1485.f9
mv CaMgO_full_tmp_1485.gui CaMgO_full_1485.gui
mv CaMgO_full_tmp_1486.f9 CaMgO_full_1486.f9
mv CaMgO_full_tmp_1486.gui CaMgO_full_1486.gui
mv CaMgO_full_tmp_1487.f9 CaMgO_full_1487.f9
mv CaMgO_full_tmp_1487.gui CaMgO_full_1487.gui
mv CaMgO_full_tmp_1488.f9 CaMgO_full_1488.f9
mv CaMgO_full_tmp_1488.gui CaMgO_full_1488.gui
mv CaMgO_full_tmp_1489.f9 CaMgO_full_1489.f9
mv CaMgO_full_tmp_1489.gui CaMgO_full_1489.gui
mv CaMgO_full_tmp_1490.f9 CaMgO_full_1490.f9
mv CaMgO_full_tmp_1490.gui CaMgO_full_1490.gui
mv CaMgO_full_tmp_1491.f9 CaMgO_full_1491.f9
mv CaMgO_full_tmp_1491.gui CaMgO_full_1491.gui
mv CaMgO_full_tmp_1492.f9 CaMgO_full_1492.f9
mv CaMgO_full_tmp_1492.gui CaMgO_full_1492.gui
mv CaMgO_full_tmp_1493.f9 CaMgO_full_1493.f9
mv CaMgO_full_tmp_1493.gui CaMgO_full_1493.gui
mv CaMgO_full_tmp_1494.f9 CaMgO_full_1494.f9
mv CaMgO_full_tmp_1494.gui CaMgO_full_1494.gui
mv CaMgO_full_tmp_1495.f9 CaMgO_full_1495.f9
mv CaMgO_full_tmp_1495.gui CaMgO_full_1495.gui
mv CaMgO_full_tmp_1496.f9 CaMgO_full_1496.f9
mv CaMgO_full_tmp_1496.gui CaMgO_full_1496.gui
mv CaMgO_full_tmp_1497.f9 CaMgO_full_1497.f9
mv CaMgO_full_tmp_1497.gui CaMgO_full_1497.gui
mv CaMgO_full_tmp_1498.f9 CaMgO_full_1498.f9
mv CaMgO_full_tmp_1498.gui CaMgO_full_1498.gui
mv CaMgO_full_tmp_1499.f9 CaMgO_full_1499.f9
mv CaMgO_full_tmp_1499.gui CaMgO_full_1499.gui
