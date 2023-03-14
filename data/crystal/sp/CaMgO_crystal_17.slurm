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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1699


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1700


