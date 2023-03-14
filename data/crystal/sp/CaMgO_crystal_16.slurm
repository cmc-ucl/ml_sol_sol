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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1501
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1501


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1502
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1502


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1503
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1503


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1504
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1504


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1505
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1505


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1506
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1506


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1507
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1507


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1508
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1508


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1509
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1509


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1510
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1510


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1511
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1511


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1512
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1512


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1513
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1513


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1514
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1514


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1515
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1515


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1516
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1516


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1517
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1517


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1518
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1518


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1519
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1519


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1520
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1520


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1521
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1521


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1522
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1522


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1523
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1523


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1524
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1524


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1525
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1525


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1526
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1526


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1527
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1527


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1528
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1528


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1529
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1529


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1530
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1530


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1531
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1531


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1532
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1532


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1533
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1533


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1534
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1534


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1535
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1535


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1536
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1536


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1537
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1537


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1538
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1538


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1539
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1539


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1540
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1540


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1541
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1541


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1542
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1542


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1543
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1543


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1544
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1544


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1545
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1545


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1546
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1546


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1547
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1547


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1548
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1548


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1549
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1549


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1550
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1550


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1551
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1551


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1552
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1552


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1553
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1553


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1554
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1554


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1555
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1555


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1556
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1556


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1557
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1557


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1558
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1558


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1559
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1559


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1560
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1560


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1561
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1561


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1562
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1562


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1563
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1563


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1564
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1564


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1565
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1565


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1566
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1566


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1567
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1567


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1568
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1568


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1569
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1569


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1570
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1570


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1571
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1571


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1572
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1572


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1573
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1573


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1574
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1574


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1575
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1575


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1576
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1576


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1577
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1577


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1578
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1578


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1579
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1579


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1580
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1580


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1581
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1581


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1582
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1582


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1583
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1583


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1584
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1584


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1585
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1585


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1586
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1586


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1587
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1587


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1588
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1588


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1589
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1589


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1590
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1590


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1591
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1591


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1592
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1592


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1593
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1593


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1594
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1594


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1595
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1595


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1596
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1596


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1597
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1597


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1598
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1598


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1599
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1599


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1600


