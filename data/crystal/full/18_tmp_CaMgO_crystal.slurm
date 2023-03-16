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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1700 CaMgO_full_tmp_1700 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1701 CaMgO_full_tmp_1701 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1702 CaMgO_full_tmp_1702 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1703 CaMgO_full_tmp_1703 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1704 CaMgO_full_tmp_1704 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1705 CaMgO_full_tmp_1705 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1706 CaMgO_full_tmp_1706 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1707 CaMgO_full_tmp_1707 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1708 CaMgO_full_tmp_1708 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1709 CaMgO_full_tmp_1709 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1710 CaMgO_full_tmp_1710 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1711 CaMgO_full_tmp_1711 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1712 CaMgO_full_tmp_1712 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1713 CaMgO_full_tmp_1713 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1714 CaMgO_full_tmp_1714 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1715 CaMgO_full_tmp_1715 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1716 CaMgO_full_tmp_1716 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1717 CaMgO_full_tmp_1717 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1718 CaMgO_full_tmp_1718 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1719 CaMgO_full_tmp_1719 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1720 CaMgO_full_tmp_1720 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1721 CaMgO_full_tmp_1721 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1722 CaMgO_full_tmp_1722 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1723 CaMgO_full_tmp_1723 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1724 CaMgO_full_tmp_1724 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1725 CaMgO_full_tmp_1725 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1726 CaMgO_full_tmp_1726 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1727 CaMgO_full_tmp_1727 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1728 CaMgO_full_tmp_1728 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1729 CaMgO_full_tmp_1729 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1730 CaMgO_full_tmp_1730 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1731 CaMgO_full_tmp_1731 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1732 CaMgO_full_tmp_1732 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1733 CaMgO_full_tmp_1733 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1734 CaMgO_full_tmp_1734 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1735 CaMgO_full_tmp_1735 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1736 CaMgO_full_tmp_1736 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1737 CaMgO_full_tmp_1737 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1738 CaMgO_full_tmp_1738 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1739 CaMgO_full_tmp_1739 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1740 CaMgO_full_tmp_1740 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1741 CaMgO_full_tmp_1741 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1742 CaMgO_full_tmp_1742 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1743 CaMgO_full_tmp_1743 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1744 CaMgO_full_tmp_1744 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1745 CaMgO_full_tmp_1745 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1746 CaMgO_full_tmp_1746 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1747 CaMgO_full_tmp_1747 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1748 CaMgO_full_tmp_1748 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1749 CaMgO_full_tmp_1749 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1750 CaMgO_full_tmp_1750 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1751 CaMgO_full_tmp_1751 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1752 CaMgO_full_tmp_1752 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1753 CaMgO_full_tmp_1753 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1754 CaMgO_full_tmp_1754 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1755 CaMgO_full_tmp_1755 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1756 CaMgO_full_tmp_1756 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1757 CaMgO_full_tmp_1757 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1758 CaMgO_full_tmp_1758 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1759 CaMgO_full_tmp_1759 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1760 CaMgO_full_tmp_1760 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1761 CaMgO_full_tmp_1761 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1762 CaMgO_full_tmp_1762 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1763 CaMgO_full_tmp_1763 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1764 CaMgO_full_tmp_1764 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1765 CaMgO_full_tmp_1765 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1766 CaMgO_full_tmp_1766 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1767 CaMgO_full_tmp_1767 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1768 CaMgO_full_tmp_1768 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1769 CaMgO_full_tmp_1769 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1770 CaMgO_full_tmp_1770 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1771 CaMgO_full_tmp_1771 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1772 CaMgO_full_tmp_1772 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1773 CaMgO_full_tmp_1773 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1774 CaMgO_full_tmp_1774 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1775 CaMgO_full_tmp_1775 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1776 CaMgO_full_tmp_1776 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1777 CaMgO_full_tmp_1777 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1778 CaMgO_full_tmp_1778 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1779 CaMgO_full_tmp_1779 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1780 CaMgO_full_tmp_1780 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1781 CaMgO_full_tmp_1781 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1782 CaMgO_full_tmp_1782 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1783 CaMgO_full_tmp_1783 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1784 CaMgO_full_tmp_1784 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1785 CaMgO_full_tmp_1785 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1786 CaMgO_full_tmp_1786 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1787 CaMgO_full_tmp_1787 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1788 CaMgO_full_tmp_1788 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1789 CaMgO_full_tmp_1789 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1790 CaMgO_full_tmp_1790 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1791 CaMgO_full_tmp_1791 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1792 CaMgO_full_tmp_1792 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1793 CaMgO_full_tmp_1793 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1794 CaMgO_full_tmp_1794 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1795 CaMgO_full_tmp_1795 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1796 CaMgO_full_tmp_1796 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1797 CaMgO_full_tmp_1797 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1798 CaMgO_full_tmp_1798 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1799 CaMgO_full_tmp_1799 &

wait
mv CaMgO_full_tmp_1700.f9 CaMgO_full_1700.f9
mv CaMgO_full_tmp_1700.gui CaMgO_full_1700.gui
mv CaMgO_full_tmp_1701.f9 CaMgO_full_1701.f9
mv CaMgO_full_tmp_1701.gui CaMgO_full_1701.gui
mv CaMgO_full_tmp_1702.f9 CaMgO_full_1702.f9
mv CaMgO_full_tmp_1702.gui CaMgO_full_1702.gui
mv CaMgO_full_tmp_1703.f9 CaMgO_full_1703.f9
mv CaMgO_full_tmp_1703.gui CaMgO_full_1703.gui
mv CaMgO_full_tmp_1704.f9 CaMgO_full_1704.f9
mv CaMgO_full_tmp_1704.gui CaMgO_full_1704.gui
mv CaMgO_full_tmp_1705.f9 CaMgO_full_1705.f9
mv CaMgO_full_tmp_1705.gui CaMgO_full_1705.gui
mv CaMgO_full_tmp_1706.f9 CaMgO_full_1706.f9
mv CaMgO_full_tmp_1706.gui CaMgO_full_1706.gui
mv CaMgO_full_tmp_1707.f9 CaMgO_full_1707.f9
mv CaMgO_full_tmp_1707.gui CaMgO_full_1707.gui
mv CaMgO_full_tmp_1708.f9 CaMgO_full_1708.f9
mv CaMgO_full_tmp_1708.gui CaMgO_full_1708.gui
mv CaMgO_full_tmp_1709.f9 CaMgO_full_1709.f9
mv CaMgO_full_tmp_1709.gui CaMgO_full_1709.gui
mv CaMgO_full_tmp_1710.f9 CaMgO_full_1710.f9
mv CaMgO_full_tmp_1710.gui CaMgO_full_1710.gui
mv CaMgO_full_tmp_1711.f9 CaMgO_full_1711.f9
mv CaMgO_full_tmp_1711.gui CaMgO_full_1711.gui
mv CaMgO_full_tmp_1712.f9 CaMgO_full_1712.f9
mv CaMgO_full_tmp_1712.gui CaMgO_full_1712.gui
mv CaMgO_full_tmp_1713.f9 CaMgO_full_1713.f9
mv CaMgO_full_tmp_1713.gui CaMgO_full_1713.gui
mv CaMgO_full_tmp_1714.f9 CaMgO_full_1714.f9
mv CaMgO_full_tmp_1714.gui CaMgO_full_1714.gui
mv CaMgO_full_tmp_1715.f9 CaMgO_full_1715.f9
mv CaMgO_full_tmp_1715.gui CaMgO_full_1715.gui
mv CaMgO_full_tmp_1716.f9 CaMgO_full_1716.f9
mv CaMgO_full_tmp_1716.gui CaMgO_full_1716.gui
mv CaMgO_full_tmp_1717.f9 CaMgO_full_1717.f9
mv CaMgO_full_tmp_1717.gui CaMgO_full_1717.gui
mv CaMgO_full_tmp_1718.f9 CaMgO_full_1718.f9
mv CaMgO_full_tmp_1718.gui CaMgO_full_1718.gui
mv CaMgO_full_tmp_1719.f9 CaMgO_full_1719.f9
mv CaMgO_full_tmp_1719.gui CaMgO_full_1719.gui
mv CaMgO_full_tmp_1720.f9 CaMgO_full_1720.f9
mv CaMgO_full_tmp_1720.gui CaMgO_full_1720.gui
mv CaMgO_full_tmp_1721.f9 CaMgO_full_1721.f9
mv CaMgO_full_tmp_1721.gui CaMgO_full_1721.gui
mv CaMgO_full_tmp_1722.f9 CaMgO_full_1722.f9
mv CaMgO_full_tmp_1722.gui CaMgO_full_1722.gui
mv CaMgO_full_tmp_1723.f9 CaMgO_full_1723.f9
mv CaMgO_full_tmp_1723.gui CaMgO_full_1723.gui
mv CaMgO_full_tmp_1724.f9 CaMgO_full_1724.f9
mv CaMgO_full_tmp_1724.gui CaMgO_full_1724.gui
mv CaMgO_full_tmp_1725.f9 CaMgO_full_1725.f9
mv CaMgO_full_tmp_1725.gui CaMgO_full_1725.gui
mv CaMgO_full_tmp_1726.f9 CaMgO_full_1726.f9
mv CaMgO_full_tmp_1726.gui CaMgO_full_1726.gui
mv CaMgO_full_tmp_1727.f9 CaMgO_full_1727.f9
mv CaMgO_full_tmp_1727.gui CaMgO_full_1727.gui
mv CaMgO_full_tmp_1728.f9 CaMgO_full_1728.f9
mv CaMgO_full_tmp_1728.gui CaMgO_full_1728.gui
mv CaMgO_full_tmp_1729.f9 CaMgO_full_1729.f9
mv CaMgO_full_tmp_1729.gui CaMgO_full_1729.gui
mv CaMgO_full_tmp_1730.f9 CaMgO_full_1730.f9
mv CaMgO_full_tmp_1730.gui CaMgO_full_1730.gui
mv CaMgO_full_tmp_1731.f9 CaMgO_full_1731.f9
mv CaMgO_full_tmp_1731.gui CaMgO_full_1731.gui
mv CaMgO_full_tmp_1732.f9 CaMgO_full_1732.f9
mv CaMgO_full_tmp_1732.gui CaMgO_full_1732.gui
mv CaMgO_full_tmp_1733.f9 CaMgO_full_1733.f9
mv CaMgO_full_tmp_1733.gui CaMgO_full_1733.gui
mv CaMgO_full_tmp_1734.f9 CaMgO_full_1734.f9
mv CaMgO_full_tmp_1734.gui CaMgO_full_1734.gui
mv CaMgO_full_tmp_1735.f9 CaMgO_full_1735.f9
mv CaMgO_full_tmp_1735.gui CaMgO_full_1735.gui
mv CaMgO_full_tmp_1736.f9 CaMgO_full_1736.f9
mv CaMgO_full_tmp_1736.gui CaMgO_full_1736.gui
mv CaMgO_full_tmp_1737.f9 CaMgO_full_1737.f9
mv CaMgO_full_tmp_1737.gui CaMgO_full_1737.gui
mv CaMgO_full_tmp_1738.f9 CaMgO_full_1738.f9
mv CaMgO_full_tmp_1738.gui CaMgO_full_1738.gui
mv CaMgO_full_tmp_1739.f9 CaMgO_full_1739.f9
mv CaMgO_full_tmp_1739.gui CaMgO_full_1739.gui
mv CaMgO_full_tmp_1740.f9 CaMgO_full_1740.f9
mv CaMgO_full_tmp_1740.gui CaMgO_full_1740.gui
mv CaMgO_full_tmp_1741.f9 CaMgO_full_1741.f9
mv CaMgO_full_tmp_1741.gui CaMgO_full_1741.gui
mv CaMgO_full_tmp_1742.f9 CaMgO_full_1742.f9
mv CaMgO_full_tmp_1742.gui CaMgO_full_1742.gui
mv CaMgO_full_tmp_1743.f9 CaMgO_full_1743.f9
mv CaMgO_full_tmp_1743.gui CaMgO_full_1743.gui
mv CaMgO_full_tmp_1744.f9 CaMgO_full_1744.f9
mv CaMgO_full_tmp_1744.gui CaMgO_full_1744.gui
mv CaMgO_full_tmp_1745.f9 CaMgO_full_1745.f9
mv CaMgO_full_tmp_1745.gui CaMgO_full_1745.gui
mv CaMgO_full_tmp_1746.f9 CaMgO_full_1746.f9
mv CaMgO_full_tmp_1746.gui CaMgO_full_1746.gui
mv CaMgO_full_tmp_1747.f9 CaMgO_full_1747.f9
mv CaMgO_full_tmp_1747.gui CaMgO_full_1747.gui
mv CaMgO_full_tmp_1748.f9 CaMgO_full_1748.f9
mv CaMgO_full_tmp_1748.gui CaMgO_full_1748.gui
mv CaMgO_full_tmp_1749.f9 CaMgO_full_1749.f9
mv CaMgO_full_tmp_1749.gui CaMgO_full_1749.gui
mv CaMgO_full_tmp_1750.f9 CaMgO_full_1750.f9
mv CaMgO_full_tmp_1750.gui CaMgO_full_1750.gui
mv CaMgO_full_tmp_1751.f9 CaMgO_full_1751.f9
mv CaMgO_full_tmp_1751.gui CaMgO_full_1751.gui
mv CaMgO_full_tmp_1752.f9 CaMgO_full_1752.f9
mv CaMgO_full_tmp_1752.gui CaMgO_full_1752.gui
mv CaMgO_full_tmp_1753.f9 CaMgO_full_1753.f9
mv CaMgO_full_tmp_1753.gui CaMgO_full_1753.gui
mv CaMgO_full_tmp_1754.f9 CaMgO_full_1754.f9
mv CaMgO_full_tmp_1754.gui CaMgO_full_1754.gui
mv CaMgO_full_tmp_1755.f9 CaMgO_full_1755.f9
mv CaMgO_full_tmp_1755.gui CaMgO_full_1755.gui
mv CaMgO_full_tmp_1756.f9 CaMgO_full_1756.f9
mv CaMgO_full_tmp_1756.gui CaMgO_full_1756.gui
mv CaMgO_full_tmp_1757.f9 CaMgO_full_1757.f9
mv CaMgO_full_tmp_1757.gui CaMgO_full_1757.gui
mv CaMgO_full_tmp_1758.f9 CaMgO_full_1758.f9
mv CaMgO_full_tmp_1758.gui CaMgO_full_1758.gui
mv CaMgO_full_tmp_1759.f9 CaMgO_full_1759.f9
mv CaMgO_full_tmp_1759.gui CaMgO_full_1759.gui
mv CaMgO_full_tmp_1760.f9 CaMgO_full_1760.f9
mv CaMgO_full_tmp_1760.gui CaMgO_full_1760.gui
mv CaMgO_full_tmp_1761.f9 CaMgO_full_1761.f9
mv CaMgO_full_tmp_1761.gui CaMgO_full_1761.gui
mv CaMgO_full_tmp_1762.f9 CaMgO_full_1762.f9
mv CaMgO_full_tmp_1762.gui CaMgO_full_1762.gui
mv CaMgO_full_tmp_1763.f9 CaMgO_full_1763.f9
mv CaMgO_full_tmp_1763.gui CaMgO_full_1763.gui
mv CaMgO_full_tmp_1764.f9 CaMgO_full_1764.f9
mv CaMgO_full_tmp_1764.gui CaMgO_full_1764.gui
mv CaMgO_full_tmp_1765.f9 CaMgO_full_1765.f9
mv CaMgO_full_tmp_1765.gui CaMgO_full_1765.gui
mv CaMgO_full_tmp_1766.f9 CaMgO_full_1766.f9
mv CaMgO_full_tmp_1766.gui CaMgO_full_1766.gui
mv CaMgO_full_tmp_1767.f9 CaMgO_full_1767.f9
mv CaMgO_full_tmp_1767.gui CaMgO_full_1767.gui
mv CaMgO_full_tmp_1768.f9 CaMgO_full_1768.f9
mv CaMgO_full_tmp_1768.gui CaMgO_full_1768.gui
mv CaMgO_full_tmp_1769.f9 CaMgO_full_1769.f9
mv CaMgO_full_tmp_1769.gui CaMgO_full_1769.gui
mv CaMgO_full_tmp_1770.f9 CaMgO_full_1770.f9
mv CaMgO_full_tmp_1770.gui CaMgO_full_1770.gui
mv CaMgO_full_tmp_1771.f9 CaMgO_full_1771.f9
mv CaMgO_full_tmp_1771.gui CaMgO_full_1771.gui
mv CaMgO_full_tmp_1772.f9 CaMgO_full_1772.f9
mv CaMgO_full_tmp_1772.gui CaMgO_full_1772.gui
mv CaMgO_full_tmp_1773.f9 CaMgO_full_1773.f9
mv CaMgO_full_tmp_1773.gui CaMgO_full_1773.gui
mv CaMgO_full_tmp_1774.f9 CaMgO_full_1774.f9
mv CaMgO_full_tmp_1774.gui CaMgO_full_1774.gui
mv CaMgO_full_tmp_1775.f9 CaMgO_full_1775.f9
mv CaMgO_full_tmp_1775.gui CaMgO_full_1775.gui
mv CaMgO_full_tmp_1776.f9 CaMgO_full_1776.f9
mv CaMgO_full_tmp_1776.gui CaMgO_full_1776.gui
mv CaMgO_full_tmp_1777.f9 CaMgO_full_1777.f9
mv CaMgO_full_tmp_1777.gui CaMgO_full_1777.gui
mv CaMgO_full_tmp_1778.f9 CaMgO_full_1778.f9
mv CaMgO_full_tmp_1778.gui CaMgO_full_1778.gui
mv CaMgO_full_tmp_1779.f9 CaMgO_full_1779.f9
mv CaMgO_full_tmp_1779.gui CaMgO_full_1779.gui
mv CaMgO_full_tmp_1780.f9 CaMgO_full_1780.f9
mv CaMgO_full_tmp_1780.gui CaMgO_full_1780.gui
mv CaMgO_full_tmp_1781.f9 CaMgO_full_1781.f9
mv CaMgO_full_tmp_1781.gui CaMgO_full_1781.gui
mv CaMgO_full_tmp_1782.f9 CaMgO_full_1782.f9
mv CaMgO_full_tmp_1782.gui CaMgO_full_1782.gui
mv CaMgO_full_tmp_1783.f9 CaMgO_full_1783.f9
mv CaMgO_full_tmp_1783.gui CaMgO_full_1783.gui
mv CaMgO_full_tmp_1784.f9 CaMgO_full_1784.f9
mv CaMgO_full_tmp_1784.gui CaMgO_full_1784.gui
mv CaMgO_full_tmp_1785.f9 CaMgO_full_1785.f9
mv CaMgO_full_tmp_1785.gui CaMgO_full_1785.gui
mv CaMgO_full_tmp_1786.f9 CaMgO_full_1786.f9
mv CaMgO_full_tmp_1786.gui CaMgO_full_1786.gui
mv CaMgO_full_tmp_1787.f9 CaMgO_full_1787.f9
mv CaMgO_full_tmp_1787.gui CaMgO_full_1787.gui
mv CaMgO_full_tmp_1788.f9 CaMgO_full_1788.f9
mv CaMgO_full_tmp_1788.gui CaMgO_full_1788.gui
mv CaMgO_full_tmp_1789.f9 CaMgO_full_1789.f9
mv CaMgO_full_tmp_1789.gui CaMgO_full_1789.gui
mv CaMgO_full_tmp_1790.f9 CaMgO_full_1790.f9
mv CaMgO_full_tmp_1790.gui CaMgO_full_1790.gui
mv CaMgO_full_tmp_1791.f9 CaMgO_full_1791.f9
mv CaMgO_full_tmp_1791.gui CaMgO_full_1791.gui
mv CaMgO_full_tmp_1792.f9 CaMgO_full_1792.f9
mv CaMgO_full_tmp_1792.gui CaMgO_full_1792.gui
mv CaMgO_full_tmp_1793.f9 CaMgO_full_1793.f9
mv CaMgO_full_tmp_1793.gui CaMgO_full_1793.gui
mv CaMgO_full_tmp_1794.f9 CaMgO_full_1794.f9
mv CaMgO_full_tmp_1794.gui CaMgO_full_1794.gui
mv CaMgO_full_tmp_1795.f9 CaMgO_full_1795.f9
mv CaMgO_full_tmp_1795.gui CaMgO_full_1795.gui
mv CaMgO_full_tmp_1796.f9 CaMgO_full_1796.f9
mv CaMgO_full_tmp_1796.gui CaMgO_full_1796.gui
mv CaMgO_full_tmp_1797.f9 CaMgO_full_1797.f9
mv CaMgO_full_tmp_1797.gui CaMgO_full_1797.gui
mv CaMgO_full_tmp_1798.f9 CaMgO_full_1798.f9
mv CaMgO_full_tmp_1798.gui CaMgO_full_1798.gui
mv CaMgO_full_tmp_1799.f9 CaMgO_full_1799.f9
mv CaMgO_full_tmp_1799.gui CaMgO_full_1799.gui
