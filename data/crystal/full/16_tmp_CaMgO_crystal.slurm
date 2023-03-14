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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1500 CaMgO_full_tmp_1500 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1501 CaMgO_full_tmp_1501 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1502 CaMgO_full_tmp_1502 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1503 CaMgO_full_tmp_1503 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1504 CaMgO_full_tmp_1504 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1505 CaMgO_full_tmp_1505 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1506 CaMgO_full_tmp_1506 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1507 CaMgO_full_tmp_1507 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1508 CaMgO_full_tmp_1508 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1509 CaMgO_full_tmp_1509 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1510 CaMgO_full_tmp_1510 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1511 CaMgO_full_tmp_1511 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1512 CaMgO_full_tmp_1512 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1513 CaMgO_full_tmp_1513 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1514 CaMgO_full_tmp_1514 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1515 CaMgO_full_tmp_1515 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1516 CaMgO_full_tmp_1516 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1517 CaMgO_full_tmp_1517 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1518 CaMgO_full_tmp_1518 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1519 CaMgO_full_tmp_1519 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1520 CaMgO_full_tmp_1520 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1521 CaMgO_full_tmp_1521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1522 CaMgO_full_tmp_1522 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1523 CaMgO_full_tmp_1523 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1524 CaMgO_full_tmp_1524 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1525 CaMgO_full_tmp_1525 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1526 CaMgO_full_tmp_1526 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1527 CaMgO_full_tmp_1527 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1528 CaMgO_full_tmp_1528 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1529 CaMgO_full_tmp_1529 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1530 CaMgO_full_tmp_1530 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1531 CaMgO_full_tmp_1531 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1532 CaMgO_full_tmp_1532 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1533 CaMgO_full_tmp_1533 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1534 CaMgO_full_tmp_1534 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1535 CaMgO_full_tmp_1535 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1536 CaMgO_full_tmp_1536 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1537 CaMgO_full_tmp_1537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1538 CaMgO_full_tmp_1538 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1539 CaMgO_full_tmp_1539 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1540 CaMgO_full_tmp_1540 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1541 CaMgO_full_tmp_1541 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1542 CaMgO_full_tmp_1542 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1543 CaMgO_full_tmp_1543 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1544 CaMgO_full_tmp_1544 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1545 CaMgO_full_tmp_1545 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1546 CaMgO_full_tmp_1546 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1547 CaMgO_full_tmp_1547 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1548 CaMgO_full_tmp_1548 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1549 CaMgO_full_tmp_1549 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1550 CaMgO_full_tmp_1550 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1551 CaMgO_full_tmp_1551 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1552 CaMgO_full_tmp_1552 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1553 CaMgO_full_tmp_1553 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1554 CaMgO_full_tmp_1554 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1555 CaMgO_full_tmp_1555 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1556 CaMgO_full_tmp_1556 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1557 CaMgO_full_tmp_1557 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1558 CaMgO_full_tmp_1558 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1559 CaMgO_full_tmp_1559 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1560 CaMgO_full_tmp_1560 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1561 CaMgO_full_tmp_1561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1562 CaMgO_full_tmp_1562 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1563 CaMgO_full_tmp_1563 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1564 CaMgO_full_tmp_1564 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1565 CaMgO_full_tmp_1565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1566 CaMgO_full_tmp_1566 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1567 CaMgO_full_tmp_1567 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1568 CaMgO_full_tmp_1568 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1569 CaMgO_full_tmp_1569 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1570 CaMgO_full_tmp_1570 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1571 CaMgO_full_tmp_1571 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1572 CaMgO_full_tmp_1572 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1573 CaMgO_full_tmp_1573 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1574 CaMgO_full_tmp_1574 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1575 CaMgO_full_tmp_1575 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1576 CaMgO_full_tmp_1576 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1577 CaMgO_full_tmp_1577 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1578 CaMgO_full_tmp_1578 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1579 CaMgO_full_tmp_1579 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1580 CaMgO_full_tmp_1580 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1581 CaMgO_full_tmp_1581 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1582 CaMgO_full_tmp_1582 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1583 CaMgO_full_tmp_1583 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1584 CaMgO_full_tmp_1584 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1585 CaMgO_full_tmp_1585 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1586 CaMgO_full_tmp_1586 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1587 CaMgO_full_tmp_1587 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1588 CaMgO_full_tmp_1588 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1589 CaMgO_full_tmp_1589 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1590 CaMgO_full_tmp_1590 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1591 CaMgO_full_tmp_1591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1592 CaMgO_full_tmp_1592 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1593 CaMgO_full_tmp_1593 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1594 CaMgO_full_tmp_1594 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1595 CaMgO_full_tmp_1595 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1596 CaMgO_full_tmp_1596 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1597 CaMgO_full_tmp_1597 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1598 CaMgO_full_tmp_1598 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1599 CaMgO_full_tmp_1599 &

wait
mv CaMgO_full_tmp_1500.f9 CaMgO_full_1500.f9
mv CaMgO_full_tmp_1500.gui CaMgO_full_1500.gui
mv CaMgO_full_tmp_1501.f9 CaMgO_full_1501.f9
mv CaMgO_full_tmp_1501.gui CaMgO_full_1501.gui
mv CaMgO_full_tmp_1502.f9 CaMgO_full_1502.f9
mv CaMgO_full_tmp_1502.gui CaMgO_full_1502.gui
mv CaMgO_full_tmp_1503.f9 CaMgO_full_1503.f9
mv CaMgO_full_tmp_1503.gui CaMgO_full_1503.gui
mv CaMgO_full_tmp_1504.f9 CaMgO_full_1504.f9
mv CaMgO_full_tmp_1504.gui CaMgO_full_1504.gui
mv CaMgO_full_tmp_1505.f9 CaMgO_full_1505.f9
mv CaMgO_full_tmp_1505.gui CaMgO_full_1505.gui
mv CaMgO_full_tmp_1506.f9 CaMgO_full_1506.f9
mv CaMgO_full_tmp_1506.gui CaMgO_full_1506.gui
mv CaMgO_full_tmp_1507.f9 CaMgO_full_1507.f9
mv CaMgO_full_tmp_1507.gui CaMgO_full_1507.gui
mv CaMgO_full_tmp_1508.f9 CaMgO_full_1508.f9
mv CaMgO_full_tmp_1508.gui CaMgO_full_1508.gui
mv CaMgO_full_tmp_1509.f9 CaMgO_full_1509.f9
mv CaMgO_full_tmp_1509.gui CaMgO_full_1509.gui
mv CaMgO_full_tmp_1510.f9 CaMgO_full_1510.f9
mv CaMgO_full_tmp_1510.gui CaMgO_full_1510.gui
mv CaMgO_full_tmp_1511.f9 CaMgO_full_1511.f9
mv CaMgO_full_tmp_1511.gui CaMgO_full_1511.gui
mv CaMgO_full_tmp_1512.f9 CaMgO_full_1512.f9
mv CaMgO_full_tmp_1512.gui CaMgO_full_1512.gui
mv CaMgO_full_tmp_1513.f9 CaMgO_full_1513.f9
mv CaMgO_full_tmp_1513.gui CaMgO_full_1513.gui
mv CaMgO_full_tmp_1514.f9 CaMgO_full_1514.f9
mv CaMgO_full_tmp_1514.gui CaMgO_full_1514.gui
mv CaMgO_full_tmp_1515.f9 CaMgO_full_1515.f9
mv CaMgO_full_tmp_1515.gui CaMgO_full_1515.gui
mv CaMgO_full_tmp_1516.f9 CaMgO_full_1516.f9
mv CaMgO_full_tmp_1516.gui CaMgO_full_1516.gui
mv CaMgO_full_tmp_1517.f9 CaMgO_full_1517.f9
mv CaMgO_full_tmp_1517.gui CaMgO_full_1517.gui
mv CaMgO_full_tmp_1518.f9 CaMgO_full_1518.f9
mv CaMgO_full_tmp_1518.gui CaMgO_full_1518.gui
mv CaMgO_full_tmp_1519.f9 CaMgO_full_1519.f9
mv CaMgO_full_tmp_1519.gui CaMgO_full_1519.gui
mv CaMgO_full_tmp_1520.f9 CaMgO_full_1520.f9
mv CaMgO_full_tmp_1520.gui CaMgO_full_1520.gui
mv CaMgO_full_tmp_1521.f9 CaMgO_full_1521.f9
mv CaMgO_full_tmp_1521.gui CaMgO_full_1521.gui
mv CaMgO_full_tmp_1522.f9 CaMgO_full_1522.f9
mv CaMgO_full_tmp_1522.gui CaMgO_full_1522.gui
mv CaMgO_full_tmp_1523.f9 CaMgO_full_1523.f9
mv CaMgO_full_tmp_1523.gui CaMgO_full_1523.gui
mv CaMgO_full_tmp_1524.f9 CaMgO_full_1524.f9
mv CaMgO_full_tmp_1524.gui CaMgO_full_1524.gui
mv CaMgO_full_tmp_1525.f9 CaMgO_full_1525.f9
mv CaMgO_full_tmp_1525.gui CaMgO_full_1525.gui
mv CaMgO_full_tmp_1526.f9 CaMgO_full_1526.f9
mv CaMgO_full_tmp_1526.gui CaMgO_full_1526.gui
mv CaMgO_full_tmp_1527.f9 CaMgO_full_1527.f9
mv CaMgO_full_tmp_1527.gui CaMgO_full_1527.gui
mv CaMgO_full_tmp_1528.f9 CaMgO_full_1528.f9
mv CaMgO_full_tmp_1528.gui CaMgO_full_1528.gui
mv CaMgO_full_tmp_1529.f9 CaMgO_full_1529.f9
mv CaMgO_full_tmp_1529.gui CaMgO_full_1529.gui
mv CaMgO_full_tmp_1530.f9 CaMgO_full_1530.f9
mv CaMgO_full_tmp_1530.gui CaMgO_full_1530.gui
mv CaMgO_full_tmp_1531.f9 CaMgO_full_1531.f9
mv CaMgO_full_tmp_1531.gui CaMgO_full_1531.gui
mv CaMgO_full_tmp_1532.f9 CaMgO_full_1532.f9
mv CaMgO_full_tmp_1532.gui CaMgO_full_1532.gui
mv CaMgO_full_tmp_1533.f9 CaMgO_full_1533.f9
mv CaMgO_full_tmp_1533.gui CaMgO_full_1533.gui
mv CaMgO_full_tmp_1534.f9 CaMgO_full_1534.f9
mv CaMgO_full_tmp_1534.gui CaMgO_full_1534.gui
mv CaMgO_full_tmp_1535.f9 CaMgO_full_1535.f9
mv CaMgO_full_tmp_1535.gui CaMgO_full_1535.gui
mv CaMgO_full_tmp_1536.f9 CaMgO_full_1536.f9
mv CaMgO_full_tmp_1536.gui CaMgO_full_1536.gui
mv CaMgO_full_tmp_1537.f9 CaMgO_full_1537.f9
mv CaMgO_full_tmp_1537.gui CaMgO_full_1537.gui
mv CaMgO_full_tmp_1538.f9 CaMgO_full_1538.f9
mv CaMgO_full_tmp_1538.gui CaMgO_full_1538.gui
mv CaMgO_full_tmp_1539.f9 CaMgO_full_1539.f9
mv CaMgO_full_tmp_1539.gui CaMgO_full_1539.gui
mv CaMgO_full_tmp_1540.f9 CaMgO_full_1540.f9
mv CaMgO_full_tmp_1540.gui CaMgO_full_1540.gui
mv CaMgO_full_tmp_1541.f9 CaMgO_full_1541.f9
mv CaMgO_full_tmp_1541.gui CaMgO_full_1541.gui
mv CaMgO_full_tmp_1542.f9 CaMgO_full_1542.f9
mv CaMgO_full_tmp_1542.gui CaMgO_full_1542.gui
mv CaMgO_full_tmp_1543.f9 CaMgO_full_1543.f9
mv CaMgO_full_tmp_1543.gui CaMgO_full_1543.gui
mv CaMgO_full_tmp_1544.f9 CaMgO_full_1544.f9
mv CaMgO_full_tmp_1544.gui CaMgO_full_1544.gui
mv CaMgO_full_tmp_1545.f9 CaMgO_full_1545.f9
mv CaMgO_full_tmp_1545.gui CaMgO_full_1545.gui
mv CaMgO_full_tmp_1546.f9 CaMgO_full_1546.f9
mv CaMgO_full_tmp_1546.gui CaMgO_full_1546.gui
mv CaMgO_full_tmp_1547.f9 CaMgO_full_1547.f9
mv CaMgO_full_tmp_1547.gui CaMgO_full_1547.gui
mv CaMgO_full_tmp_1548.f9 CaMgO_full_1548.f9
mv CaMgO_full_tmp_1548.gui CaMgO_full_1548.gui
mv CaMgO_full_tmp_1549.f9 CaMgO_full_1549.f9
mv CaMgO_full_tmp_1549.gui CaMgO_full_1549.gui
mv CaMgO_full_tmp_1550.f9 CaMgO_full_1550.f9
mv CaMgO_full_tmp_1550.gui CaMgO_full_1550.gui
mv CaMgO_full_tmp_1551.f9 CaMgO_full_1551.f9
mv CaMgO_full_tmp_1551.gui CaMgO_full_1551.gui
mv CaMgO_full_tmp_1552.f9 CaMgO_full_1552.f9
mv CaMgO_full_tmp_1552.gui CaMgO_full_1552.gui
mv CaMgO_full_tmp_1553.f9 CaMgO_full_1553.f9
mv CaMgO_full_tmp_1553.gui CaMgO_full_1553.gui
mv CaMgO_full_tmp_1554.f9 CaMgO_full_1554.f9
mv CaMgO_full_tmp_1554.gui CaMgO_full_1554.gui
mv CaMgO_full_tmp_1555.f9 CaMgO_full_1555.f9
mv CaMgO_full_tmp_1555.gui CaMgO_full_1555.gui
mv CaMgO_full_tmp_1556.f9 CaMgO_full_1556.f9
mv CaMgO_full_tmp_1556.gui CaMgO_full_1556.gui
mv CaMgO_full_tmp_1557.f9 CaMgO_full_1557.f9
mv CaMgO_full_tmp_1557.gui CaMgO_full_1557.gui
mv CaMgO_full_tmp_1558.f9 CaMgO_full_1558.f9
mv CaMgO_full_tmp_1558.gui CaMgO_full_1558.gui
mv CaMgO_full_tmp_1559.f9 CaMgO_full_1559.f9
mv CaMgO_full_tmp_1559.gui CaMgO_full_1559.gui
mv CaMgO_full_tmp_1560.f9 CaMgO_full_1560.f9
mv CaMgO_full_tmp_1560.gui CaMgO_full_1560.gui
mv CaMgO_full_tmp_1561.f9 CaMgO_full_1561.f9
mv CaMgO_full_tmp_1561.gui CaMgO_full_1561.gui
mv CaMgO_full_tmp_1562.f9 CaMgO_full_1562.f9
mv CaMgO_full_tmp_1562.gui CaMgO_full_1562.gui
mv CaMgO_full_tmp_1563.f9 CaMgO_full_1563.f9
mv CaMgO_full_tmp_1563.gui CaMgO_full_1563.gui
mv CaMgO_full_tmp_1564.f9 CaMgO_full_1564.f9
mv CaMgO_full_tmp_1564.gui CaMgO_full_1564.gui
mv CaMgO_full_tmp_1565.f9 CaMgO_full_1565.f9
mv CaMgO_full_tmp_1565.gui CaMgO_full_1565.gui
mv CaMgO_full_tmp_1566.f9 CaMgO_full_1566.f9
mv CaMgO_full_tmp_1566.gui CaMgO_full_1566.gui
mv CaMgO_full_tmp_1567.f9 CaMgO_full_1567.f9
mv CaMgO_full_tmp_1567.gui CaMgO_full_1567.gui
mv CaMgO_full_tmp_1568.f9 CaMgO_full_1568.f9
mv CaMgO_full_tmp_1568.gui CaMgO_full_1568.gui
mv CaMgO_full_tmp_1569.f9 CaMgO_full_1569.f9
mv CaMgO_full_tmp_1569.gui CaMgO_full_1569.gui
mv CaMgO_full_tmp_1570.f9 CaMgO_full_1570.f9
mv CaMgO_full_tmp_1570.gui CaMgO_full_1570.gui
mv CaMgO_full_tmp_1571.f9 CaMgO_full_1571.f9
mv CaMgO_full_tmp_1571.gui CaMgO_full_1571.gui
mv CaMgO_full_tmp_1572.f9 CaMgO_full_1572.f9
mv CaMgO_full_tmp_1572.gui CaMgO_full_1572.gui
mv CaMgO_full_tmp_1573.f9 CaMgO_full_1573.f9
mv CaMgO_full_tmp_1573.gui CaMgO_full_1573.gui
mv CaMgO_full_tmp_1574.f9 CaMgO_full_1574.f9
mv CaMgO_full_tmp_1574.gui CaMgO_full_1574.gui
mv CaMgO_full_tmp_1575.f9 CaMgO_full_1575.f9
mv CaMgO_full_tmp_1575.gui CaMgO_full_1575.gui
mv CaMgO_full_tmp_1576.f9 CaMgO_full_1576.f9
mv CaMgO_full_tmp_1576.gui CaMgO_full_1576.gui
mv CaMgO_full_tmp_1577.f9 CaMgO_full_1577.f9
mv CaMgO_full_tmp_1577.gui CaMgO_full_1577.gui
mv CaMgO_full_tmp_1578.f9 CaMgO_full_1578.f9
mv CaMgO_full_tmp_1578.gui CaMgO_full_1578.gui
mv CaMgO_full_tmp_1579.f9 CaMgO_full_1579.f9
mv CaMgO_full_tmp_1579.gui CaMgO_full_1579.gui
mv CaMgO_full_tmp_1580.f9 CaMgO_full_1580.f9
mv CaMgO_full_tmp_1580.gui CaMgO_full_1580.gui
mv CaMgO_full_tmp_1581.f9 CaMgO_full_1581.f9
mv CaMgO_full_tmp_1581.gui CaMgO_full_1581.gui
mv CaMgO_full_tmp_1582.f9 CaMgO_full_1582.f9
mv CaMgO_full_tmp_1582.gui CaMgO_full_1582.gui
mv CaMgO_full_tmp_1583.f9 CaMgO_full_1583.f9
mv CaMgO_full_tmp_1583.gui CaMgO_full_1583.gui
mv CaMgO_full_tmp_1584.f9 CaMgO_full_1584.f9
mv CaMgO_full_tmp_1584.gui CaMgO_full_1584.gui
mv CaMgO_full_tmp_1585.f9 CaMgO_full_1585.f9
mv CaMgO_full_tmp_1585.gui CaMgO_full_1585.gui
mv CaMgO_full_tmp_1586.f9 CaMgO_full_1586.f9
mv CaMgO_full_tmp_1586.gui CaMgO_full_1586.gui
mv CaMgO_full_tmp_1587.f9 CaMgO_full_1587.f9
mv CaMgO_full_tmp_1587.gui CaMgO_full_1587.gui
mv CaMgO_full_tmp_1588.f9 CaMgO_full_1588.f9
mv CaMgO_full_tmp_1588.gui CaMgO_full_1588.gui
mv CaMgO_full_tmp_1589.f9 CaMgO_full_1589.f9
mv CaMgO_full_tmp_1589.gui CaMgO_full_1589.gui
mv CaMgO_full_tmp_1590.f9 CaMgO_full_1590.f9
mv CaMgO_full_tmp_1590.gui CaMgO_full_1590.gui
mv CaMgO_full_tmp_1591.f9 CaMgO_full_1591.f9
mv CaMgO_full_tmp_1591.gui CaMgO_full_1591.gui
mv CaMgO_full_tmp_1592.f9 CaMgO_full_1592.f9
mv CaMgO_full_tmp_1592.gui CaMgO_full_1592.gui
mv CaMgO_full_tmp_1593.f9 CaMgO_full_1593.f9
mv CaMgO_full_tmp_1593.gui CaMgO_full_1593.gui
mv CaMgO_full_tmp_1594.f9 CaMgO_full_1594.f9
mv CaMgO_full_tmp_1594.gui CaMgO_full_1594.gui
mv CaMgO_full_tmp_1595.f9 CaMgO_full_1595.f9
mv CaMgO_full_tmp_1595.gui CaMgO_full_1595.gui
mv CaMgO_full_tmp_1596.f9 CaMgO_full_1596.f9
mv CaMgO_full_tmp_1596.gui CaMgO_full_1596.gui
mv CaMgO_full_tmp_1597.f9 CaMgO_full_1597.f9
mv CaMgO_full_tmp_1597.gui CaMgO_full_1597.gui
mv CaMgO_full_tmp_1598.f9 CaMgO_full_1598.f9
mv CaMgO_full_tmp_1598.gui CaMgO_full_1598.gui
mv CaMgO_full_tmp_1599.f9 CaMgO_full_1599.f9
mv CaMgO_full_tmp_1599.gui CaMgO_full_1599.gui
