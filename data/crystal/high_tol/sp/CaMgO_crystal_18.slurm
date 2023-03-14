#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-gc-smw    
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1700


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1701
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1701


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1702
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1702


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1703
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1703


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1704
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1704


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1705
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1705


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1706
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1706


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1707
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1707


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1708
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1708


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1709
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1709


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1710
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1710


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1711
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1711


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1712
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1712


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1713
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1713


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1714
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1714


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1715
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1715


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1716
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1716


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1717
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1717


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1718
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1718


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1719
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1719


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1720
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1720


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1721
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1721


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1722
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1722


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1723
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1723


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1724
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1724


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1725
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1725


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1726
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1726


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1727
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1727


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1728
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1728


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1729
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1729


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1730
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1730


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1731
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1731


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1732
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1732


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1733
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1733


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1734
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1734


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1735
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1735


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1736
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1736


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1737
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1737


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1738
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1738


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1739
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1739


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1740
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1740


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1741
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1741


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1742
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1742


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1743
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1743


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1744
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1744


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1745
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1745


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1746
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1746


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1747
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1747


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1748
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1748


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1749
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1749


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1750
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1750


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1751
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1751


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1752
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1752


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1753
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1753


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1754
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1754


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1755
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1755


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1756
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1756


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1757
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1757


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1758
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1758


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1759
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1759


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1760
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1760


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1761
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1761


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1762
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1762


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1763
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1763


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1764
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1764


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1765
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1765


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1766
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1766


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1767
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1767


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1768
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1768


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1769
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1769


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1770
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1770


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1771
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1771


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1772
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1772


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1773
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1773


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1774
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1774


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1775
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1775


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1776
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1776


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1777
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1777


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1778
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1778


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1779
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1779


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1780
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1780


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1781
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1781


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1782
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1782


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1783
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1783


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1784
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1784


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1785
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1785


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1786
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1786


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1787
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1787


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1788
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1788


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1789
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1789


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1790
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1790


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1791
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1791


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1792
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1792


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1793
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1793


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1794
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1794


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1795
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1795


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1796
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1796


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1797
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1797


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1798
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1798


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1799
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1799


