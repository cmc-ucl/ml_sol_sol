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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1600 CaMgO_full_tmp_1600 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1601 CaMgO_full_tmp_1601 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1602 CaMgO_full_tmp_1602 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1603 CaMgO_full_tmp_1603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1604 CaMgO_full_tmp_1604 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1605 CaMgO_full_tmp_1605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1606 CaMgO_full_tmp_1606 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1607 CaMgO_full_tmp_1607 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1608 CaMgO_full_tmp_1608 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1609 CaMgO_full_tmp_1609 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1610 CaMgO_full_tmp_1610 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1611 CaMgO_full_tmp_1611 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1612 CaMgO_full_tmp_1612 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1613 CaMgO_full_tmp_1613 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1614 CaMgO_full_tmp_1614 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1615 CaMgO_full_tmp_1615 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1616 CaMgO_full_tmp_1616 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1617 CaMgO_full_tmp_1617 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1618 CaMgO_full_tmp_1618 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1619 CaMgO_full_tmp_1619 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1620 CaMgO_full_tmp_1620 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1621 CaMgO_full_tmp_1621 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1622 CaMgO_full_tmp_1622 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1623 CaMgO_full_tmp_1623 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1624 CaMgO_full_tmp_1624 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1625 CaMgO_full_tmp_1625 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1626 CaMgO_full_tmp_1626 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1627 CaMgO_full_tmp_1627 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1628 CaMgO_full_tmp_1628 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1629 CaMgO_full_tmp_1629 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1630 CaMgO_full_tmp_1630 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1631 CaMgO_full_tmp_1631 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1632 CaMgO_full_tmp_1632 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1633 CaMgO_full_tmp_1633 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1634 CaMgO_full_tmp_1634 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1635 CaMgO_full_tmp_1635 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1636 CaMgO_full_tmp_1636 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1637 CaMgO_full_tmp_1637 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1638 CaMgO_full_tmp_1638 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1639 CaMgO_full_tmp_1639 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1640 CaMgO_full_tmp_1640 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1641 CaMgO_full_tmp_1641 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1642 CaMgO_full_tmp_1642 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1643 CaMgO_full_tmp_1643 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1644 CaMgO_full_tmp_1644 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1645 CaMgO_full_tmp_1645 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1646 CaMgO_full_tmp_1646 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1647 CaMgO_full_tmp_1647 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1648 CaMgO_full_tmp_1648 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1649 CaMgO_full_tmp_1649 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1650 CaMgO_full_tmp_1650 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1651 CaMgO_full_tmp_1651 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1652 CaMgO_full_tmp_1652 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1653 CaMgO_full_tmp_1653 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1654 CaMgO_full_tmp_1654 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1655 CaMgO_full_tmp_1655 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1656 CaMgO_full_tmp_1656 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1657 CaMgO_full_tmp_1657 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1658 CaMgO_full_tmp_1658 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1659 CaMgO_full_tmp_1659 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1660 CaMgO_full_tmp_1660 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1661 CaMgO_full_tmp_1661 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1662 CaMgO_full_tmp_1662 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1663 CaMgO_full_tmp_1663 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1664 CaMgO_full_tmp_1664 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1665 CaMgO_full_tmp_1665 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1666 CaMgO_full_tmp_1666 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1667 CaMgO_full_tmp_1667 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1668 CaMgO_full_tmp_1668 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1669 CaMgO_full_tmp_1669 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1670 CaMgO_full_tmp_1670 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1671 CaMgO_full_tmp_1671 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1672 CaMgO_full_tmp_1672 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1673 CaMgO_full_tmp_1673 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1674 CaMgO_full_tmp_1674 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1675 CaMgO_full_tmp_1675 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1676 CaMgO_full_tmp_1676 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1677 CaMgO_full_tmp_1677 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1678 CaMgO_full_tmp_1678 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1679 CaMgO_full_tmp_1679 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1680 CaMgO_full_tmp_1680 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1681 CaMgO_full_tmp_1681 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1682 CaMgO_full_tmp_1682 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1683 CaMgO_full_tmp_1683 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1684 CaMgO_full_tmp_1684 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1685 CaMgO_full_tmp_1685 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1686 CaMgO_full_tmp_1686 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1687 CaMgO_full_tmp_1687 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1688 CaMgO_full_tmp_1688 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1689 CaMgO_full_tmp_1689 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1690 CaMgO_full_tmp_1690 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1691 CaMgO_full_tmp_1691 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1692 CaMgO_full_tmp_1692 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1693 CaMgO_full_tmp_1693 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1694 CaMgO_full_tmp_1694 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1695 CaMgO_full_tmp_1695 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1696 CaMgO_full_tmp_1696 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1697 CaMgO_full_tmp_1697 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1698 CaMgO_full_tmp_1698 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1699 CaMgO_full_tmp_1699 &

wait
mv CaMgO_full_tmp_1600.f9 CaMgO_full_1600.f9
mv CaMgO_full_tmp_1600.gui CaMgO_full_1600.gui
mv CaMgO_full_tmp_1601.f9 CaMgO_full_1601.f9
mv CaMgO_full_tmp_1601.gui CaMgO_full_1601.gui
mv CaMgO_full_tmp_1602.f9 CaMgO_full_1602.f9
mv CaMgO_full_tmp_1602.gui CaMgO_full_1602.gui
mv CaMgO_full_tmp_1603.f9 CaMgO_full_1603.f9
mv CaMgO_full_tmp_1603.gui CaMgO_full_1603.gui
mv CaMgO_full_tmp_1604.f9 CaMgO_full_1604.f9
mv CaMgO_full_tmp_1604.gui CaMgO_full_1604.gui
mv CaMgO_full_tmp_1605.f9 CaMgO_full_1605.f9
mv CaMgO_full_tmp_1605.gui CaMgO_full_1605.gui
mv CaMgO_full_tmp_1606.f9 CaMgO_full_1606.f9
mv CaMgO_full_tmp_1606.gui CaMgO_full_1606.gui
mv CaMgO_full_tmp_1607.f9 CaMgO_full_1607.f9
mv CaMgO_full_tmp_1607.gui CaMgO_full_1607.gui
mv CaMgO_full_tmp_1608.f9 CaMgO_full_1608.f9
mv CaMgO_full_tmp_1608.gui CaMgO_full_1608.gui
mv CaMgO_full_tmp_1609.f9 CaMgO_full_1609.f9
mv CaMgO_full_tmp_1609.gui CaMgO_full_1609.gui
mv CaMgO_full_tmp_1610.f9 CaMgO_full_1610.f9
mv CaMgO_full_tmp_1610.gui CaMgO_full_1610.gui
mv CaMgO_full_tmp_1611.f9 CaMgO_full_1611.f9
mv CaMgO_full_tmp_1611.gui CaMgO_full_1611.gui
mv CaMgO_full_tmp_1612.f9 CaMgO_full_1612.f9
mv CaMgO_full_tmp_1612.gui CaMgO_full_1612.gui
mv CaMgO_full_tmp_1613.f9 CaMgO_full_1613.f9
mv CaMgO_full_tmp_1613.gui CaMgO_full_1613.gui
mv CaMgO_full_tmp_1614.f9 CaMgO_full_1614.f9
mv CaMgO_full_tmp_1614.gui CaMgO_full_1614.gui
mv CaMgO_full_tmp_1615.f9 CaMgO_full_1615.f9
mv CaMgO_full_tmp_1615.gui CaMgO_full_1615.gui
mv CaMgO_full_tmp_1616.f9 CaMgO_full_1616.f9
mv CaMgO_full_tmp_1616.gui CaMgO_full_1616.gui
mv CaMgO_full_tmp_1617.f9 CaMgO_full_1617.f9
mv CaMgO_full_tmp_1617.gui CaMgO_full_1617.gui
mv CaMgO_full_tmp_1618.f9 CaMgO_full_1618.f9
mv CaMgO_full_tmp_1618.gui CaMgO_full_1618.gui
mv CaMgO_full_tmp_1619.f9 CaMgO_full_1619.f9
mv CaMgO_full_tmp_1619.gui CaMgO_full_1619.gui
mv CaMgO_full_tmp_1620.f9 CaMgO_full_1620.f9
mv CaMgO_full_tmp_1620.gui CaMgO_full_1620.gui
mv CaMgO_full_tmp_1621.f9 CaMgO_full_1621.f9
mv CaMgO_full_tmp_1621.gui CaMgO_full_1621.gui
mv CaMgO_full_tmp_1622.f9 CaMgO_full_1622.f9
mv CaMgO_full_tmp_1622.gui CaMgO_full_1622.gui
mv CaMgO_full_tmp_1623.f9 CaMgO_full_1623.f9
mv CaMgO_full_tmp_1623.gui CaMgO_full_1623.gui
mv CaMgO_full_tmp_1624.f9 CaMgO_full_1624.f9
mv CaMgO_full_tmp_1624.gui CaMgO_full_1624.gui
mv CaMgO_full_tmp_1625.f9 CaMgO_full_1625.f9
mv CaMgO_full_tmp_1625.gui CaMgO_full_1625.gui
mv CaMgO_full_tmp_1626.f9 CaMgO_full_1626.f9
mv CaMgO_full_tmp_1626.gui CaMgO_full_1626.gui
mv CaMgO_full_tmp_1627.f9 CaMgO_full_1627.f9
mv CaMgO_full_tmp_1627.gui CaMgO_full_1627.gui
mv CaMgO_full_tmp_1628.f9 CaMgO_full_1628.f9
mv CaMgO_full_tmp_1628.gui CaMgO_full_1628.gui
mv CaMgO_full_tmp_1629.f9 CaMgO_full_1629.f9
mv CaMgO_full_tmp_1629.gui CaMgO_full_1629.gui
mv CaMgO_full_tmp_1630.f9 CaMgO_full_1630.f9
mv CaMgO_full_tmp_1630.gui CaMgO_full_1630.gui
mv CaMgO_full_tmp_1631.f9 CaMgO_full_1631.f9
mv CaMgO_full_tmp_1631.gui CaMgO_full_1631.gui
mv CaMgO_full_tmp_1632.f9 CaMgO_full_1632.f9
mv CaMgO_full_tmp_1632.gui CaMgO_full_1632.gui
mv CaMgO_full_tmp_1633.f9 CaMgO_full_1633.f9
mv CaMgO_full_tmp_1633.gui CaMgO_full_1633.gui
mv CaMgO_full_tmp_1634.f9 CaMgO_full_1634.f9
mv CaMgO_full_tmp_1634.gui CaMgO_full_1634.gui
mv CaMgO_full_tmp_1635.f9 CaMgO_full_1635.f9
mv CaMgO_full_tmp_1635.gui CaMgO_full_1635.gui
mv CaMgO_full_tmp_1636.f9 CaMgO_full_1636.f9
mv CaMgO_full_tmp_1636.gui CaMgO_full_1636.gui
mv CaMgO_full_tmp_1637.f9 CaMgO_full_1637.f9
mv CaMgO_full_tmp_1637.gui CaMgO_full_1637.gui
mv CaMgO_full_tmp_1638.f9 CaMgO_full_1638.f9
mv CaMgO_full_tmp_1638.gui CaMgO_full_1638.gui
mv CaMgO_full_tmp_1639.f9 CaMgO_full_1639.f9
mv CaMgO_full_tmp_1639.gui CaMgO_full_1639.gui
mv CaMgO_full_tmp_1640.f9 CaMgO_full_1640.f9
mv CaMgO_full_tmp_1640.gui CaMgO_full_1640.gui
mv CaMgO_full_tmp_1641.f9 CaMgO_full_1641.f9
mv CaMgO_full_tmp_1641.gui CaMgO_full_1641.gui
mv CaMgO_full_tmp_1642.f9 CaMgO_full_1642.f9
mv CaMgO_full_tmp_1642.gui CaMgO_full_1642.gui
mv CaMgO_full_tmp_1643.f9 CaMgO_full_1643.f9
mv CaMgO_full_tmp_1643.gui CaMgO_full_1643.gui
mv CaMgO_full_tmp_1644.f9 CaMgO_full_1644.f9
mv CaMgO_full_tmp_1644.gui CaMgO_full_1644.gui
mv CaMgO_full_tmp_1645.f9 CaMgO_full_1645.f9
mv CaMgO_full_tmp_1645.gui CaMgO_full_1645.gui
mv CaMgO_full_tmp_1646.f9 CaMgO_full_1646.f9
mv CaMgO_full_tmp_1646.gui CaMgO_full_1646.gui
mv CaMgO_full_tmp_1647.f9 CaMgO_full_1647.f9
mv CaMgO_full_tmp_1647.gui CaMgO_full_1647.gui
mv CaMgO_full_tmp_1648.f9 CaMgO_full_1648.f9
mv CaMgO_full_tmp_1648.gui CaMgO_full_1648.gui
mv CaMgO_full_tmp_1649.f9 CaMgO_full_1649.f9
mv CaMgO_full_tmp_1649.gui CaMgO_full_1649.gui
mv CaMgO_full_tmp_1650.f9 CaMgO_full_1650.f9
mv CaMgO_full_tmp_1650.gui CaMgO_full_1650.gui
mv CaMgO_full_tmp_1651.f9 CaMgO_full_1651.f9
mv CaMgO_full_tmp_1651.gui CaMgO_full_1651.gui
mv CaMgO_full_tmp_1652.f9 CaMgO_full_1652.f9
mv CaMgO_full_tmp_1652.gui CaMgO_full_1652.gui
mv CaMgO_full_tmp_1653.f9 CaMgO_full_1653.f9
mv CaMgO_full_tmp_1653.gui CaMgO_full_1653.gui
mv CaMgO_full_tmp_1654.f9 CaMgO_full_1654.f9
mv CaMgO_full_tmp_1654.gui CaMgO_full_1654.gui
mv CaMgO_full_tmp_1655.f9 CaMgO_full_1655.f9
mv CaMgO_full_tmp_1655.gui CaMgO_full_1655.gui
mv CaMgO_full_tmp_1656.f9 CaMgO_full_1656.f9
mv CaMgO_full_tmp_1656.gui CaMgO_full_1656.gui
mv CaMgO_full_tmp_1657.f9 CaMgO_full_1657.f9
mv CaMgO_full_tmp_1657.gui CaMgO_full_1657.gui
mv CaMgO_full_tmp_1658.f9 CaMgO_full_1658.f9
mv CaMgO_full_tmp_1658.gui CaMgO_full_1658.gui
mv CaMgO_full_tmp_1659.f9 CaMgO_full_1659.f9
mv CaMgO_full_tmp_1659.gui CaMgO_full_1659.gui
mv CaMgO_full_tmp_1660.f9 CaMgO_full_1660.f9
mv CaMgO_full_tmp_1660.gui CaMgO_full_1660.gui
mv CaMgO_full_tmp_1661.f9 CaMgO_full_1661.f9
mv CaMgO_full_tmp_1661.gui CaMgO_full_1661.gui
mv CaMgO_full_tmp_1662.f9 CaMgO_full_1662.f9
mv CaMgO_full_tmp_1662.gui CaMgO_full_1662.gui
mv CaMgO_full_tmp_1663.f9 CaMgO_full_1663.f9
mv CaMgO_full_tmp_1663.gui CaMgO_full_1663.gui
mv CaMgO_full_tmp_1664.f9 CaMgO_full_1664.f9
mv CaMgO_full_tmp_1664.gui CaMgO_full_1664.gui
mv CaMgO_full_tmp_1665.f9 CaMgO_full_1665.f9
mv CaMgO_full_tmp_1665.gui CaMgO_full_1665.gui
mv CaMgO_full_tmp_1666.f9 CaMgO_full_1666.f9
mv CaMgO_full_tmp_1666.gui CaMgO_full_1666.gui
mv CaMgO_full_tmp_1667.f9 CaMgO_full_1667.f9
mv CaMgO_full_tmp_1667.gui CaMgO_full_1667.gui
mv CaMgO_full_tmp_1668.f9 CaMgO_full_1668.f9
mv CaMgO_full_tmp_1668.gui CaMgO_full_1668.gui
mv CaMgO_full_tmp_1669.f9 CaMgO_full_1669.f9
mv CaMgO_full_tmp_1669.gui CaMgO_full_1669.gui
mv CaMgO_full_tmp_1670.f9 CaMgO_full_1670.f9
mv CaMgO_full_tmp_1670.gui CaMgO_full_1670.gui
mv CaMgO_full_tmp_1671.f9 CaMgO_full_1671.f9
mv CaMgO_full_tmp_1671.gui CaMgO_full_1671.gui
mv CaMgO_full_tmp_1672.f9 CaMgO_full_1672.f9
mv CaMgO_full_tmp_1672.gui CaMgO_full_1672.gui
mv CaMgO_full_tmp_1673.f9 CaMgO_full_1673.f9
mv CaMgO_full_tmp_1673.gui CaMgO_full_1673.gui
mv CaMgO_full_tmp_1674.f9 CaMgO_full_1674.f9
mv CaMgO_full_tmp_1674.gui CaMgO_full_1674.gui
mv CaMgO_full_tmp_1675.f9 CaMgO_full_1675.f9
mv CaMgO_full_tmp_1675.gui CaMgO_full_1675.gui
mv CaMgO_full_tmp_1676.f9 CaMgO_full_1676.f9
mv CaMgO_full_tmp_1676.gui CaMgO_full_1676.gui
mv CaMgO_full_tmp_1677.f9 CaMgO_full_1677.f9
mv CaMgO_full_tmp_1677.gui CaMgO_full_1677.gui
mv CaMgO_full_tmp_1678.f9 CaMgO_full_1678.f9
mv CaMgO_full_tmp_1678.gui CaMgO_full_1678.gui
mv CaMgO_full_tmp_1679.f9 CaMgO_full_1679.f9
mv CaMgO_full_tmp_1679.gui CaMgO_full_1679.gui
mv CaMgO_full_tmp_1680.f9 CaMgO_full_1680.f9
mv CaMgO_full_tmp_1680.gui CaMgO_full_1680.gui
mv CaMgO_full_tmp_1681.f9 CaMgO_full_1681.f9
mv CaMgO_full_tmp_1681.gui CaMgO_full_1681.gui
mv CaMgO_full_tmp_1682.f9 CaMgO_full_1682.f9
mv CaMgO_full_tmp_1682.gui CaMgO_full_1682.gui
mv CaMgO_full_tmp_1683.f9 CaMgO_full_1683.f9
mv CaMgO_full_tmp_1683.gui CaMgO_full_1683.gui
mv CaMgO_full_tmp_1684.f9 CaMgO_full_1684.f9
mv CaMgO_full_tmp_1684.gui CaMgO_full_1684.gui
mv CaMgO_full_tmp_1685.f9 CaMgO_full_1685.f9
mv CaMgO_full_tmp_1685.gui CaMgO_full_1685.gui
mv CaMgO_full_tmp_1686.f9 CaMgO_full_1686.f9
mv CaMgO_full_tmp_1686.gui CaMgO_full_1686.gui
mv CaMgO_full_tmp_1687.f9 CaMgO_full_1687.f9
mv CaMgO_full_tmp_1687.gui CaMgO_full_1687.gui
mv CaMgO_full_tmp_1688.f9 CaMgO_full_1688.f9
mv CaMgO_full_tmp_1688.gui CaMgO_full_1688.gui
mv CaMgO_full_tmp_1689.f9 CaMgO_full_1689.f9
mv CaMgO_full_tmp_1689.gui CaMgO_full_1689.gui
mv CaMgO_full_tmp_1690.f9 CaMgO_full_1690.f9
mv CaMgO_full_tmp_1690.gui CaMgO_full_1690.gui
mv CaMgO_full_tmp_1691.f9 CaMgO_full_1691.f9
mv CaMgO_full_tmp_1691.gui CaMgO_full_1691.gui
mv CaMgO_full_tmp_1692.f9 CaMgO_full_1692.f9
mv CaMgO_full_tmp_1692.gui CaMgO_full_1692.gui
mv CaMgO_full_tmp_1693.f9 CaMgO_full_1693.f9
mv CaMgO_full_tmp_1693.gui CaMgO_full_1693.gui
mv CaMgO_full_tmp_1694.f9 CaMgO_full_1694.f9
mv CaMgO_full_tmp_1694.gui CaMgO_full_1694.gui
mv CaMgO_full_tmp_1695.f9 CaMgO_full_1695.f9
mv CaMgO_full_tmp_1695.gui CaMgO_full_1695.gui
mv CaMgO_full_tmp_1696.f9 CaMgO_full_1696.f9
mv CaMgO_full_tmp_1696.gui CaMgO_full_1696.gui
mv CaMgO_full_tmp_1697.f9 CaMgO_full_1697.f9
mv CaMgO_full_tmp_1697.gui CaMgO_full_1697.gui
mv CaMgO_full_tmp_1698.f9 CaMgO_full_1698.f9
mv CaMgO_full_tmp_1698.gui CaMgO_full_1698.gui
mv CaMgO_full_tmp_1699.f9 CaMgO_full_1699.f9
mv CaMgO_full_tmp_1699.gui CaMgO_full_1699.gui
