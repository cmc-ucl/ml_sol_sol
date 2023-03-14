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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1800
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1800


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1899


