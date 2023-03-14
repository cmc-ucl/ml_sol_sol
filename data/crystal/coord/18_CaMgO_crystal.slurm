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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1701 CaMgO_coord_1701
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1701


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1702 CaMgO_coord_1702
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1702


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1703 CaMgO_coord_1703
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1703


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1704 CaMgO_coord_1704
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1704


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1705 CaMgO_coord_1705
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1705


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1706 CaMgO_coord_1706
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1706


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1707 CaMgO_coord_1707
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1707


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1708 CaMgO_coord_1708
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1708


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1709 CaMgO_coord_1709
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1709


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1710 CaMgO_coord_1710
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1710


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1711 CaMgO_coord_1711
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1711


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1712 CaMgO_coord_1712
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1712


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1713 CaMgO_coord_1713
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1713


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1714 CaMgO_coord_1714
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1714


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1715 CaMgO_coord_1715
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1715


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1716 CaMgO_coord_1716
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1716


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1717 CaMgO_coord_1717
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1717


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1718 CaMgO_coord_1718
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1718


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1719 CaMgO_coord_1719
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1719


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1720 CaMgO_coord_1720
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1720


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1721 CaMgO_coord_1721
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1721


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1722 CaMgO_coord_1722
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1722


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1723 CaMgO_coord_1723
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1723


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1724 CaMgO_coord_1724
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1724


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1725 CaMgO_coord_1725
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1725


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1726 CaMgO_coord_1726
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1726


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1727 CaMgO_coord_1727
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1727


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1728 CaMgO_coord_1728
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1728


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1729 CaMgO_coord_1729
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1729


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1730 CaMgO_coord_1730
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1730


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1731 CaMgO_coord_1731
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1731


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1732 CaMgO_coord_1732
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1732


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1733 CaMgO_coord_1733
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1733


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1734 CaMgO_coord_1734
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1734


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1735 CaMgO_coord_1735
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1735


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1736 CaMgO_coord_1736
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1736


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1737 CaMgO_coord_1737
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1737


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1738 CaMgO_coord_1738
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1738


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1739 CaMgO_coord_1739
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1739


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1740 CaMgO_coord_1740
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1740


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1741 CaMgO_coord_1741
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1741


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1742 CaMgO_coord_1742
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1742


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1743 CaMgO_coord_1743
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1743


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1744 CaMgO_coord_1744
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1744


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1745 CaMgO_coord_1745
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1745


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1746 CaMgO_coord_1746
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1746


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1747 CaMgO_coord_1747
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1747


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1748 CaMgO_coord_1748
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1748


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1749 CaMgO_coord_1749
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1749


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1750 CaMgO_coord_1750
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1750


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1751 CaMgO_coord_1751
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1751


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1752 CaMgO_coord_1752
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1752


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1753 CaMgO_coord_1753
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1753


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1754 CaMgO_coord_1754
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1754


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1755 CaMgO_coord_1755
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1755


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1756 CaMgO_coord_1756
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1756


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1757 CaMgO_coord_1757
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1757


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1758 CaMgO_coord_1758
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1758


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1759 CaMgO_coord_1759
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1759


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1760 CaMgO_coord_1760
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1760


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1761 CaMgO_coord_1761
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1761


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1762 CaMgO_coord_1762
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1762


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1763 CaMgO_coord_1763
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1763


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1764 CaMgO_coord_1764
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1764


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1765 CaMgO_coord_1765
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1765


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1766 CaMgO_coord_1766
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1766


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1767 CaMgO_coord_1767
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1767


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1768 CaMgO_coord_1768
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1768


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1769 CaMgO_coord_1769
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1769


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1770 CaMgO_coord_1770
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1770


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1771 CaMgO_coord_1771
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1771


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1772 CaMgO_coord_1772
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1772


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1773 CaMgO_coord_1773
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1773


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1774 CaMgO_coord_1774
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1774


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1775 CaMgO_coord_1775
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1775


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1776 CaMgO_coord_1776
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1776


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1777 CaMgO_coord_1777
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1777


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1778 CaMgO_coord_1778
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1778


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1779 CaMgO_coord_1779
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1779


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1780 CaMgO_coord_1780
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1780


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1781 CaMgO_coord_1781
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1781


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1782 CaMgO_coord_1782
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1782


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1783 CaMgO_coord_1783
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1783


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1784 CaMgO_coord_1784
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1784


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1785 CaMgO_coord_1785
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1785


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1786 CaMgO_coord_1786
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1786


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1787 CaMgO_coord_1787
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1787


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1788 CaMgO_coord_1788
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1788


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1789 CaMgO_coord_1789
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1789


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1790 CaMgO_coord_1790
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1790


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1791 CaMgO_coord_1791
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1791


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1792 CaMgO_coord_1792
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1792


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1793 CaMgO_coord_1793
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1793


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1794 CaMgO_coord_1794
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1794


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1795 CaMgO_coord_1795
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1795


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1796 CaMgO_coord_1796
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1796


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1797 CaMgO_coord_1797
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1797


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1798 CaMgO_coord_1798
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1798


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1799 CaMgO_coord_1799
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1799


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1800 CaMgO_coord_1800
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1800


