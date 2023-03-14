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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1800 CaMgO_full_tmp_1800 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1801 CaMgO_full_tmp_1801 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1802 CaMgO_full_tmp_1802 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1803 CaMgO_full_tmp_1803 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1804 CaMgO_full_tmp_1804 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1805 CaMgO_full_tmp_1805 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1806 CaMgO_full_tmp_1806 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1807 CaMgO_full_tmp_1807 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1808 CaMgO_full_tmp_1808 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1809 CaMgO_full_tmp_1809 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1810 CaMgO_full_tmp_1810 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1811 CaMgO_full_tmp_1811 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1812 CaMgO_full_tmp_1812 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1813 CaMgO_full_tmp_1813 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1814 CaMgO_full_tmp_1814 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1815 CaMgO_full_tmp_1815 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1816 CaMgO_full_tmp_1816 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1817 CaMgO_full_tmp_1817 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1818 CaMgO_full_tmp_1818 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1819 CaMgO_full_tmp_1819 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1820 CaMgO_full_tmp_1820 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1821 CaMgO_full_tmp_1821 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1822 CaMgO_full_tmp_1822 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1823 CaMgO_full_tmp_1823 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1824 CaMgO_full_tmp_1824 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1825 CaMgO_full_tmp_1825 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1826 CaMgO_full_tmp_1826 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1827 CaMgO_full_tmp_1827 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1828 CaMgO_full_tmp_1828 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1829 CaMgO_full_tmp_1829 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1830 CaMgO_full_tmp_1830 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1831 CaMgO_full_tmp_1831 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1832 CaMgO_full_tmp_1832 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1833 CaMgO_full_tmp_1833 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1834 CaMgO_full_tmp_1834 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1835 CaMgO_full_tmp_1835 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1836 CaMgO_full_tmp_1836 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1837 CaMgO_full_tmp_1837 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1838 CaMgO_full_tmp_1838 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1839 CaMgO_full_tmp_1839 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1840 CaMgO_full_tmp_1840 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1841 CaMgO_full_tmp_1841 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1842 CaMgO_full_tmp_1842 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1843 CaMgO_full_tmp_1843 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1844 CaMgO_full_tmp_1844 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1845 CaMgO_full_tmp_1845 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1846 CaMgO_full_tmp_1846 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1847 CaMgO_full_tmp_1847 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1848 CaMgO_full_tmp_1848 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1849 CaMgO_full_tmp_1849 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1850 CaMgO_full_tmp_1850 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1851 CaMgO_full_tmp_1851 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1852 CaMgO_full_tmp_1852 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1853 CaMgO_full_tmp_1853 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1854 CaMgO_full_tmp_1854 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1855 CaMgO_full_tmp_1855 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1856 CaMgO_full_tmp_1856 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1857 CaMgO_full_tmp_1857 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1858 CaMgO_full_tmp_1858 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1859 CaMgO_full_tmp_1859 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1860 CaMgO_full_tmp_1860 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1861 CaMgO_full_tmp_1861 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1862 CaMgO_full_tmp_1862 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1863 CaMgO_full_tmp_1863 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1864 CaMgO_full_tmp_1864 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1865 CaMgO_full_tmp_1865 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1866 CaMgO_full_tmp_1866 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1867 CaMgO_full_tmp_1867 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1868 CaMgO_full_tmp_1868 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1869 CaMgO_full_tmp_1869 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1870 CaMgO_full_tmp_1870 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1871 CaMgO_full_tmp_1871 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1872 CaMgO_full_tmp_1872 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1873 CaMgO_full_tmp_1873 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1874 CaMgO_full_tmp_1874 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1875 CaMgO_full_tmp_1875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1876 CaMgO_full_tmp_1876 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1877 CaMgO_full_tmp_1877 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1878 CaMgO_full_tmp_1878 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1879 CaMgO_full_tmp_1879 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1880 CaMgO_full_tmp_1880 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1881 CaMgO_full_tmp_1881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1882 CaMgO_full_tmp_1882 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1883 CaMgO_full_tmp_1883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1884 CaMgO_full_tmp_1884 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1885 CaMgO_full_tmp_1885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1886 CaMgO_full_tmp_1886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1887 CaMgO_full_tmp_1887 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1888 CaMgO_full_tmp_1888 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1889 CaMgO_full_tmp_1889 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1890 CaMgO_full_tmp_1890 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1891 CaMgO_full_tmp_1891 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1892 CaMgO_full_tmp_1892 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1893 CaMgO_full_tmp_1893 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1894 CaMgO_full_tmp_1894 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1895 CaMgO_full_tmp_1895 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1896 CaMgO_full_tmp_1896 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1897 CaMgO_full_tmp_1897 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1898 CaMgO_full_tmp_1898 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1899 CaMgO_full_tmp_1899 &

wait
mv CaMgO_full_tmp_1800.f9 CaMgO_full_1800.f9
mv CaMgO_full_tmp_1800.gui CaMgO_full_1800.gui
mv CaMgO_full_tmp_1801.f9 CaMgO_full_1801.f9
mv CaMgO_full_tmp_1801.gui CaMgO_full_1801.gui
mv CaMgO_full_tmp_1802.f9 CaMgO_full_1802.f9
mv CaMgO_full_tmp_1802.gui CaMgO_full_1802.gui
mv CaMgO_full_tmp_1803.f9 CaMgO_full_1803.f9
mv CaMgO_full_tmp_1803.gui CaMgO_full_1803.gui
mv CaMgO_full_tmp_1804.f9 CaMgO_full_1804.f9
mv CaMgO_full_tmp_1804.gui CaMgO_full_1804.gui
mv CaMgO_full_tmp_1805.f9 CaMgO_full_1805.f9
mv CaMgO_full_tmp_1805.gui CaMgO_full_1805.gui
mv CaMgO_full_tmp_1806.f9 CaMgO_full_1806.f9
mv CaMgO_full_tmp_1806.gui CaMgO_full_1806.gui
mv CaMgO_full_tmp_1807.f9 CaMgO_full_1807.f9
mv CaMgO_full_tmp_1807.gui CaMgO_full_1807.gui
mv CaMgO_full_tmp_1808.f9 CaMgO_full_1808.f9
mv CaMgO_full_tmp_1808.gui CaMgO_full_1808.gui
mv CaMgO_full_tmp_1809.f9 CaMgO_full_1809.f9
mv CaMgO_full_tmp_1809.gui CaMgO_full_1809.gui
mv CaMgO_full_tmp_1810.f9 CaMgO_full_1810.f9
mv CaMgO_full_tmp_1810.gui CaMgO_full_1810.gui
mv CaMgO_full_tmp_1811.f9 CaMgO_full_1811.f9
mv CaMgO_full_tmp_1811.gui CaMgO_full_1811.gui
mv CaMgO_full_tmp_1812.f9 CaMgO_full_1812.f9
mv CaMgO_full_tmp_1812.gui CaMgO_full_1812.gui
mv CaMgO_full_tmp_1813.f9 CaMgO_full_1813.f9
mv CaMgO_full_tmp_1813.gui CaMgO_full_1813.gui
mv CaMgO_full_tmp_1814.f9 CaMgO_full_1814.f9
mv CaMgO_full_tmp_1814.gui CaMgO_full_1814.gui
mv CaMgO_full_tmp_1815.f9 CaMgO_full_1815.f9
mv CaMgO_full_tmp_1815.gui CaMgO_full_1815.gui
mv CaMgO_full_tmp_1816.f9 CaMgO_full_1816.f9
mv CaMgO_full_tmp_1816.gui CaMgO_full_1816.gui
mv CaMgO_full_tmp_1817.f9 CaMgO_full_1817.f9
mv CaMgO_full_tmp_1817.gui CaMgO_full_1817.gui
mv CaMgO_full_tmp_1818.f9 CaMgO_full_1818.f9
mv CaMgO_full_tmp_1818.gui CaMgO_full_1818.gui
mv CaMgO_full_tmp_1819.f9 CaMgO_full_1819.f9
mv CaMgO_full_tmp_1819.gui CaMgO_full_1819.gui
mv CaMgO_full_tmp_1820.f9 CaMgO_full_1820.f9
mv CaMgO_full_tmp_1820.gui CaMgO_full_1820.gui
mv CaMgO_full_tmp_1821.f9 CaMgO_full_1821.f9
mv CaMgO_full_tmp_1821.gui CaMgO_full_1821.gui
mv CaMgO_full_tmp_1822.f9 CaMgO_full_1822.f9
mv CaMgO_full_tmp_1822.gui CaMgO_full_1822.gui
mv CaMgO_full_tmp_1823.f9 CaMgO_full_1823.f9
mv CaMgO_full_tmp_1823.gui CaMgO_full_1823.gui
mv CaMgO_full_tmp_1824.f9 CaMgO_full_1824.f9
mv CaMgO_full_tmp_1824.gui CaMgO_full_1824.gui
mv CaMgO_full_tmp_1825.f9 CaMgO_full_1825.f9
mv CaMgO_full_tmp_1825.gui CaMgO_full_1825.gui
mv CaMgO_full_tmp_1826.f9 CaMgO_full_1826.f9
mv CaMgO_full_tmp_1826.gui CaMgO_full_1826.gui
mv CaMgO_full_tmp_1827.f9 CaMgO_full_1827.f9
mv CaMgO_full_tmp_1827.gui CaMgO_full_1827.gui
mv CaMgO_full_tmp_1828.f9 CaMgO_full_1828.f9
mv CaMgO_full_tmp_1828.gui CaMgO_full_1828.gui
mv CaMgO_full_tmp_1829.f9 CaMgO_full_1829.f9
mv CaMgO_full_tmp_1829.gui CaMgO_full_1829.gui
mv CaMgO_full_tmp_1830.f9 CaMgO_full_1830.f9
mv CaMgO_full_tmp_1830.gui CaMgO_full_1830.gui
mv CaMgO_full_tmp_1831.f9 CaMgO_full_1831.f9
mv CaMgO_full_tmp_1831.gui CaMgO_full_1831.gui
mv CaMgO_full_tmp_1832.f9 CaMgO_full_1832.f9
mv CaMgO_full_tmp_1832.gui CaMgO_full_1832.gui
mv CaMgO_full_tmp_1833.f9 CaMgO_full_1833.f9
mv CaMgO_full_tmp_1833.gui CaMgO_full_1833.gui
mv CaMgO_full_tmp_1834.f9 CaMgO_full_1834.f9
mv CaMgO_full_tmp_1834.gui CaMgO_full_1834.gui
mv CaMgO_full_tmp_1835.f9 CaMgO_full_1835.f9
mv CaMgO_full_tmp_1835.gui CaMgO_full_1835.gui
mv CaMgO_full_tmp_1836.f9 CaMgO_full_1836.f9
mv CaMgO_full_tmp_1836.gui CaMgO_full_1836.gui
mv CaMgO_full_tmp_1837.f9 CaMgO_full_1837.f9
mv CaMgO_full_tmp_1837.gui CaMgO_full_1837.gui
mv CaMgO_full_tmp_1838.f9 CaMgO_full_1838.f9
mv CaMgO_full_tmp_1838.gui CaMgO_full_1838.gui
mv CaMgO_full_tmp_1839.f9 CaMgO_full_1839.f9
mv CaMgO_full_tmp_1839.gui CaMgO_full_1839.gui
mv CaMgO_full_tmp_1840.f9 CaMgO_full_1840.f9
mv CaMgO_full_tmp_1840.gui CaMgO_full_1840.gui
mv CaMgO_full_tmp_1841.f9 CaMgO_full_1841.f9
mv CaMgO_full_tmp_1841.gui CaMgO_full_1841.gui
mv CaMgO_full_tmp_1842.f9 CaMgO_full_1842.f9
mv CaMgO_full_tmp_1842.gui CaMgO_full_1842.gui
mv CaMgO_full_tmp_1843.f9 CaMgO_full_1843.f9
mv CaMgO_full_tmp_1843.gui CaMgO_full_1843.gui
mv CaMgO_full_tmp_1844.f9 CaMgO_full_1844.f9
mv CaMgO_full_tmp_1844.gui CaMgO_full_1844.gui
mv CaMgO_full_tmp_1845.f9 CaMgO_full_1845.f9
mv CaMgO_full_tmp_1845.gui CaMgO_full_1845.gui
mv CaMgO_full_tmp_1846.f9 CaMgO_full_1846.f9
mv CaMgO_full_tmp_1846.gui CaMgO_full_1846.gui
mv CaMgO_full_tmp_1847.f9 CaMgO_full_1847.f9
mv CaMgO_full_tmp_1847.gui CaMgO_full_1847.gui
mv CaMgO_full_tmp_1848.f9 CaMgO_full_1848.f9
mv CaMgO_full_tmp_1848.gui CaMgO_full_1848.gui
mv CaMgO_full_tmp_1849.f9 CaMgO_full_1849.f9
mv CaMgO_full_tmp_1849.gui CaMgO_full_1849.gui
mv CaMgO_full_tmp_1850.f9 CaMgO_full_1850.f9
mv CaMgO_full_tmp_1850.gui CaMgO_full_1850.gui
mv CaMgO_full_tmp_1851.f9 CaMgO_full_1851.f9
mv CaMgO_full_tmp_1851.gui CaMgO_full_1851.gui
mv CaMgO_full_tmp_1852.f9 CaMgO_full_1852.f9
mv CaMgO_full_tmp_1852.gui CaMgO_full_1852.gui
mv CaMgO_full_tmp_1853.f9 CaMgO_full_1853.f9
mv CaMgO_full_tmp_1853.gui CaMgO_full_1853.gui
mv CaMgO_full_tmp_1854.f9 CaMgO_full_1854.f9
mv CaMgO_full_tmp_1854.gui CaMgO_full_1854.gui
mv CaMgO_full_tmp_1855.f9 CaMgO_full_1855.f9
mv CaMgO_full_tmp_1855.gui CaMgO_full_1855.gui
mv CaMgO_full_tmp_1856.f9 CaMgO_full_1856.f9
mv CaMgO_full_tmp_1856.gui CaMgO_full_1856.gui
mv CaMgO_full_tmp_1857.f9 CaMgO_full_1857.f9
mv CaMgO_full_tmp_1857.gui CaMgO_full_1857.gui
mv CaMgO_full_tmp_1858.f9 CaMgO_full_1858.f9
mv CaMgO_full_tmp_1858.gui CaMgO_full_1858.gui
mv CaMgO_full_tmp_1859.f9 CaMgO_full_1859.f9
mv CaMgO_full_tmp_1859.gui CaMgO_full_1859.gui
mv CaMgO_full_tmp_1860.f9 CaMgO_full_1860.f9
mv CaMgO_full_tmp_1860.gui CaMgO_full_1860.gui
mv CaMgO_full_tmp_1861.f9 CaMgO_full_1861.f9
mv CaMgO_full_tmp_1861.gui CaMgO_full_1861.gui
mv CaMgO_full_tmp_1862.f9 CaMgO_full_1862.f9
mv CaMgO_full_tmp_1862.gui CaMgO_full_1862.gui
mv CaMgO_full_tmp_1863.f9 CaMgO_full_1863.f9
mv CaMgO_full_tmp_1863.gui CaMgO_full_1863.gui
mv CaMgO_full_tmp_1864.f9 CaMgO_full_1864.f9
mv CaMgO_full_tmp_1864.gui CaMgO_full_1864.gui
mv CaMgO_full_tmp_1865.f9 CaMgO_full_1865.f9
mv CaMgO_full_tmp_1865.gui CaMgO_full_1865.gui
mv CaMgO_full_tmp_1866.f9 CaMgO_full_1866.f9
mv CaMgO_full_tmp_1866.gui CaMgO_full_1866.gui
mv CaMgO_full_tmp_1867.f9 CaMgO_full_1867.f9
mv CaMgO_full_tmp_1867.gui CaMgO_full_1867.gui
mv CaMgO_full_tmp_1868.f9 CaMgO_full_1868.f9
mv CaMgO_full_tmp_1868.gui CaMgO_full_1868.gui
mv CaMgO_full_tmp_1869.f9 CaMgO_full_1869.f9
mv CaMgO_full_tmp_1869.gui CaMgO_full_1869.gui
mv CaMgO_full_tmp_1870.f9 CaMgO_full_1870.f9
mv CaMgO_full_tmp_1870.gui CaMgO_full_1870.gui
mv CaMgO_full_tmp_1871.f9 CaMgO_full_1871.f9
mv CaMgO_full_tmp_1871.gui CaMgO_full_1871.gui
mv CaMgO_full_tmp_1872.f9 CaMgO_full_1872.f9
mv CaMgO_full_tmp_1872.gui CaMgO_full_1872.gui
mv CaMgO_full_tmp_1873.f9 CaMgO_full_1873.f9
mv CaMgO_full_tmp_1873.gui CaMgO_full_1873.gui
mv CaMgO_full_tmp_1874.f9 CaMgO_full_1874.f9
mv CaMgO_full_tmp_1874.gui CaMgO_full_1874.gui
mv CaMgO_full_tmp_1875.f9 CaMgO_full_1875.f9
mv CaMgO_full_tmp_1875.gui CaMgO_full_1875.gui
mv CaMgO_full_tmp_1876.f9 CaMgO_full_1876.f9
mv CaMgO_full_tmp_1876.gui CaMgO_full_1876.gui
mv CaMgO_full_tmp_1877.f9 CaMgO_full_1877.f9
mv CaMgO_full_tmp_1877.gui CaMgO_full_1877.gui
mv CaMgO_full_tmp_1878.f9 CaMgO_full_1878.f9
mv CaMgO_full_tmp_1878.gui CaMgO_full_1878.gui
mv CaMgO_full_tmp_1879.f9 CaMgO_full_1879.f9
mv CaMgO_full_tmp_1879.gui CaMgO_full_1879.gui
mv CaMgO_full_tmp_1880.f9 CaMgO_full_1880.f9
mv CaMgO_full_tmp_1880.gui CaMgO_full_1880.gui
mv CaMgO_full_tmp_1881.f9 CaMgO_full_1881.f9
mv CaMgO_full_tmp_1881.gui CaMgO_full_1881.gui
mv CaMgO_full_tmp_1882.f9 CaMgO_full_1882.f9
mv CaMgO_full_tmp_1882.gui CaMgO_full_1882.gui
mv CaMgO_full_tmp_1883.f9 CaMgO_full_1883.f9
mv CaMgO_full_tmp_1883.gui CaMgO_full_1883.gui
mv CaMgO_full_tmp_1884.f9 CaMgO_full_1884.f9
mv CaMgO_full_tmp_1884.gui CaMgO_full_1884.gui
mv CaMgO_full_tmp_1885.f9 CaMgO_full_1885.f9
mv CaMgO_full_tmp_1885.gui CaMgO_full_1885.gui
mv CaMgO_full_tmp_1886.f9 CaMgO_full_1886.f9
mv CaMgO_full_tmp_1886.gui CaMgO_full_1886.gui
mv CaMgO_full_tmp_1887.f9 CaMgO_full_1887.f9
mv CaMgO_full_tmp_1887.gui CaMgO_full_1887.gui
mv CaMgO_full_tmp_1888.f9 CaMgO_full_1888.f9
mv CaMgO_full_tmp_1888.gui CaMgO_full_1888.gui
mv CaMgO_full_tmp_1889.f9 CaMgO_full_1889.f9
mv CaMgO_full_tmp_1889.gui CaMgO_full_1889.gui
mv CaMgO_full_tmp_1890.f9 CaMgO_full_1890.f9
mv CaMgO_full_tmp_1890.gui CaMgO_full_1890.gui
mv CaMgO_full_tmp_1891.f9 CaMgO_full_1891.f9
mv CaMgO_full_tmp_1891.gui CaMgO_full_1891.gui
mv CaMgO_full_tmp_1892.f9 CaMgO_full_1892.f9
mv CaMgO_full_tmp_1892.gui CaMgO_full_1892.gui
mv CaMgO_full_tmp_1893.f9 CaMgO_full_1893.f9
mv CaMgO_full_tmp_1893.gui CaMgO_full_1893.gui
mv CaMgO_full_tmp_1894.f9 CaMgO_full_1894.f9
mv CaMgO_full_tmp_1894.gui CaMgO_full_1894.gui
mv CaMgO_full_tmp_1895.f9 CaMgO_full_1895.f9
mv CaMgO_full_tmp_1895.gui CaMgO_full_1895.gui
mv CaMgO_full_tmp_1896.f9 CaMgO_full_1896.f9
mv CaMgO_full_tmp_1896.gui CaMgO_full_1896.gui
mv CaMgO_full_tmp_1897.f9 CaMgO_full_1897.f9
mv CaMgO_full_tmp_1897.gui CaMgO_full_1897.gui
mv CaMgO_full_tmp_1898.f9 CaMgO_full_1898.f9
mv CaMgO_full_tmp_1898.gui CaMgO_full_1898.gui
mv CaMgO_full_tmp_1899.f9 CaMgO_full_1899.f9
mv CaMgO_full_tmp_1899.gui CaMgO_full_1899.gui
