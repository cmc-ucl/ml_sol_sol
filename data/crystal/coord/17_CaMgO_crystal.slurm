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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1601 CaMgO_coord_1601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1602 CaMgO_coord_1602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1603 CaMgO_coord_1603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1604 CaMgO_coord_1604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1605 CaMgO_coord_1605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1606 CaMgO_coord_1606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1607 CaMgO_coord_1607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1608 CaMgO_coord_1608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1609 CaMgO_coord_1609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1610 CaMgO_coord_1610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1611 CaMgO_coord_1611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1612 CaMgO_coord_1612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1613 CaMgO_coord_1613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1614 CaMgO_coord_1614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1615 CaMgO_coord_1615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1616 CaMgO_coord_1616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1617 CaMgO_coord_1617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1618 CaMgO_coord_1618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1619 CaMgO_coord_1619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1620 CaMgO_coord_1620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1621 CaMgO_coord_1621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1622 CaMgO_coord_1622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1623 CaMgO_coord_1623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1624 CaMgO_coord_1624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1625 CaMgO_coord_1625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1626 CaMgO_coord_1626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1627 CaMgO_coord_1627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1628 CaMgO_coord_1628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1629 CaMgO_coord_1629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1630 CaMgO_coord_1630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1631 CaMgO_coord_1631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1632 CaMgO_coord_1632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1633 CaMgO_coord_1633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1634 CaMgO_coord_1634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1635 CaMgO_coord_1635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1636 CaMgO_coord_1636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1637 CaMgO_coord_1637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1638 CaMgO_coord_1638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1639 CaMgO_coord_1639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1640 CaMgO_coord_1640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1641 CaMgO_coord_1641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1642 CaMgO_coord_1642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1643 CaMgO_coord_1643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1644 CaMgO_coord_1644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1645 CaMgO_coord_1645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1646 CaMgO_coord_1646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1647 CaMgO_coord_1647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1648 CaMgO_coord_1648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1649 CaMgO_coord_1649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1650 CaMgO_coord_1650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1651 CaMgO_coord_1651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1652 CaMgO_coord_1652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1653 CaMgO_coord_1653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1654 CaMgO_coord_1654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1655 CaMgO_coord_1655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1656 CaMgO_coord_1656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1657 CaMgO_coord_1657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1658 CaMgO_coord_1658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1659 CaMgO_coord_1659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1660 CaMgO_coord_1660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1661 CaMgO_coord_1661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1662 CaMgO_coord_1662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1663 CaMgO_coord_1663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1664 CaMgO_coord_1664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1665 CaMgO_coord_1665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1666 CaMgO_coord_1666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1667 CaMgO_coord_1667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1668 CaMgO_coord_1668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1669 CaMgO_coord_1669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1670 CaMgO_coord_1670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1671 CaMgO_coord_1671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1672 CaMgO_coord_1672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1673 CaMgO_coord_1673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1674 CaMgO_coord_1674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1675 CaMgO_coord_1675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1676 CaMgO_coord_1676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1677 CaMgO_coord_1677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1678 CaMgO_coord_1678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1679 CaMgO_coord_1679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1680 CaMgO_coord_1680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1681 CaMgO_coord_1681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1682 CaMgO_coord_1682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1683 CaMgO_coord_1683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1684 CaMgO_coord_1684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1685 CaMgO_coord_1685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1686 CaMgO_coord_1686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1687 CaMgO_coord_1687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1688 CaMgO_coord_1688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1689 CaMgO_coord_1689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1690 CaMgO_coord_1690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1691 CaMgO_coord_1691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1692 CaMgO_coord_1692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1693 CaMgO_coord_1693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1694 CaMgO_coord_1694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1695 CaMgO_coord_1695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1696 CaMgO_coord_1696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1697 CaMgO_coord_1697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1698 CaMgO_coord_1698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1699 CaMgO_coord_1699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1699


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1700 CaMgO_coord_1700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1700


