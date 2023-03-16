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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1900


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1999


