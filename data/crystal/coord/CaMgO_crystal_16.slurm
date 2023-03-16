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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1501 CaMgO_coord_tmp_1501
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1501


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1502 CaMgO_coord_tmp_1502
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1502


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1503 CaMgO_coord_tmp_1503
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1503


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1504 CaMgO_coord_tmp_1504
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1504


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1505 CaMgO_coord_tmp_1505
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1505


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1506 CaMgO_coord_tmp_1506
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1506


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1507 CaMgO_coord_tmp_1507
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1507


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1508 CaMgO_coord_tmp_1508
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1508


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1509 CaMgO_coord_tmp_1509
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1509


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1510 CaMgO_coord_tmp_1510
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1510


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1511 CaMgO_coord_tmp_1511
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1511


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1512 CaMgO_coord_tmp_1512
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1512


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1513 CaMgO_coord_tmp_1513
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1513


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1514 CaMgO_coord_tmp_1514
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1514


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1515 CaMgO_coord_tmp_1515
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1515


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1516 CaMgO_coord_tmp_1516
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1516


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1517 CaMgO_coord_tmp_1517
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1517


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1518 CaMgO_coord_tmp_1518
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1518


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1519 CaMgO_coord_tmp_1519
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1519


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1520 CaMgO_coord_tmp_1520
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1520


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1521 CaMgO_coord_tmp_1521
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1521


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1522 CaMgO_coord_tmp_1522
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1522


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1523 CaMgO_coord_tmp_1523
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1523


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1524 CaMgO_coord_tmp_1524
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1524


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1525 CaMgO_coord_tmp_1525
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1525


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1526 CaMgO_coord_tmp_1526
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1526


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1527 CaMgO_coord_tmp_1527
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1527


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1528 CaMgO_coord_tmp_1528
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1528


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1529 CaMgO_coord_tmp_1529
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1529


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1530 CaMgO_coord_tmp_1530
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1530


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1531 CaMgO_coord_tmp_1531
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1531


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1532 CaMgO_coord_tmp_1532
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1532


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1533 CaMgO_coord_tmp_1533
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1533


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1534 CaMgO_coord_tmp_1534
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1534


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1535 CaMgO_coord_tmp_1535
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1535


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1536 CaMgO_coord_tmp_1536
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1536


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1537 CaMgO_coord_tmp_1537
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1537


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1538 CaMgO_coord_tmp_1538
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1538


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1539 CaMgO_coord_tmp_1539
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1539


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1540 CaMgO_coord_tmp_1540
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1540


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1541 CaMgO_coord_tmp_1541
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1541


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1542 CaMgO_coord_tmp_1542
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1542


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1543 CaMgO_coord_tmp_1543
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1543


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1544 CaMgO_coord_tmp_1544
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1544


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1545 CaMgO_coord_tmp_1545
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1545


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1546 CaMgO_coord_tmp_1546
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1546


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1547 CaMgO_coord_tmp_1547
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1547


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1548 CaMgO_coord_tmp_1548
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1548


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1549 CaMgO_coord_tmp_1549
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1549


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1550 CaMgO_coord_tmp_1550
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1550


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1551 CaMgO_coord_tmp_1551
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1551


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1552 CaMgO_coord_tmp_1552
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1552


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1553 CaMgO_coord_tmp_1553
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1553


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1554 CaMgO_coord_tmp_1554
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1554


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1555 CaMgO_coord_tmp_1555
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1555


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1556 CaMgO_coord_tmp_1556
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1556


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1557 CaMgO_coord_tmp_1557
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1557


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1558 CaMgO_coord_tmp_1558
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1558


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1559 CaMgO_coord_tmp_1559
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1559


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1560 CaMgO_coord_tmp_1560
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1560


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1561 CaMgO_coord_tmp_1561
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1561


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1562 CaMgO_coord_tmp_1562
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1562


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1563 CaMgO_coord_tmp_1563
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1563


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1564 CaMgO_coord_tmp_1564
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1564


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1565 CaMgO_coord_tmp_1565
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1565


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1566 CaMgO_coord_tmp_1566
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1566


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1567 CaMgO_coord_tmp_1567
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1567


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1568 CaMgO_coord_tmp_1568
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1568


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1569 CaMgO_coord_tmp_1569
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1569


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1570 CaMgO_coord_tmp_1570
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1570


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1571 CaMgO_coord_tmp_1571
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1571


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1572 CaMgO_coord_tmp_1572
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1572


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1573 CaMgO_coord_tmp_1573
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1573


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1574 CaMgO_coord_tmp_1574
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1574


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1575 CaMgO_coord_tmp_1575
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1575


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1576 CaMgO_coord_tmp_1576
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1576


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1577 CaMgO_coord_tmp_1577
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1577


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1578 CaMgO_coord_tmp_1578
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1578


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1579 CaMgO_coord_tmp_1579
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1579


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1580 CaMgO_coord_tmp_1580
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1580


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1581 CaMgO_coord_tmp_1581
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1581


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1582 CaMgO_coord_tmp_1582
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1582


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1583 CaMgO_coord_tmp_1583
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1583


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1584 CaMgO_coord_tmp_1584
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1584


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1585 CaMgO_coord_tmp_1585
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1585


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1586 CaMgO_coord_tmp_1586
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1586


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1587 CaMgO_coord_tmp_1587
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1587


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1588 CaMgO_coord_tmp_1588
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1588


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1589 CaMgO_coord_tmp_1589
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1589


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1590 CaMgO_coord_tmp_1590
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1590


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1591 CaMgO_coord_tmp_1591
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1591


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1592 CaMgO_coord_tmp_1592
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1592


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1593 CaMgO_coord_tmp_1593
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1593


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1594 CaMgO_coord_tmp_1594
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1594


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1595 CaMgO_coord_tmp_1595
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1595


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1596 CaMgO_coord_tmp_1596
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1596


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1597 CaMgO_coord_tmp_1597
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1597


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1598 CaMgO_coord_tmp_1598
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1598


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1599 CaMgO_coord_tmp_1599
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1599


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_1600 CaMgO_coord_tmp_1600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_1600


