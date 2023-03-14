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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1900 CaMgO_full_tmp_1900 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1901 CaMgO_full_tmp_1901 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1902 CaMgO_full_tmp_1902 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1903 CaMgO_full_tmp_1903 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1904 CaMgO_full_tmp_1904 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1905 CaMgO_full_tmp_1905 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1906 CaMgO_full_tmp_1906 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1907 CaMgO_full_tmp_1907 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1908 CaMgO_full_tmp_1908 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1909 CaMgO_full_tmp_1909 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1910 CaMgO_full_tmp_1910 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1911 CaMgO_full_tmp_1911 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1912 CaMgO_full_tmp_1912 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1913 CaMgO_full_tmp_1913 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1914 CaMgO_full_tmp_1914 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1915 CaMgO_full_tmp_1915 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1916 CaMgO_full_tmp_1916 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1917 CaMgO_full_tmp_1917 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1918 CaMgO_full_tmp_1918 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1919 CaMgO_full_tmp_1919 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1920 CaMgO_full_tmp_1920 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1921 CaMgO_full_tmp_1921 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1922 CaMgO_full_tmp_1922 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1923 CaMgO_full_tmp_1923 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1924 CaMgO_full_tmp_1924 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1925 CaMgO_full_tmp_1925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1926 CaMgO_full_tmp_1926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1927 CaMgO_full_tmp_1927 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1928 CaMgO_full_tmp_1928 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1929 CaMgO_full_tmp_1929 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1930 CaMgO_full_tmp_1930 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1931 CaMgO_full_tmp_1931 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1932 CaMgO_full_tmp_1932 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1933 CaMgO_full_tmp_1933 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1934 CaMgO_full_tmp_1934 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1935 CaMgO_full_tmp_1935 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1936 CaMgO_full_tmp_1936 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1937 CaMgO_full_tmp_1937 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1938 CaMgO_full_tmp_1938 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1939 CaMgO_full_tmp_1939 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1940 CaMgO_full_tmp_1940 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1941 CaMgO_full_tmp_1941 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1942 CaMgO_full_tmp_1942 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1943 CaMgO_full_tmp_1943 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1944 CaMgO_full_tmp_1944 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1945 CaMgO_full_tmp_1945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1946 CaMgO_full_tmp_1946 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1947 CaMgO_full_tmp_1947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1948 CaMgO_full_tmp_1948 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1949 CaMgO_full_tmp_1949 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1950 CaMgO_full_tmp_1950 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1951 CaMgO_full_tmp_1951 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1952 CaMgO_full_tmp_1952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1953 CaMgO_full_tmp_1953 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1954 CaMgO_full_tmp_1954 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1955 CaMgO_full_tmp_1955 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1956 CaMgO_full_tmp_1956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1957 CaMgO_full_tmp_1957 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1958 CaMgO_full_tmp_1958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1959 CaMgO_full_tmp_1959 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1960 CaMgO_full_tmp_1960 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1961 CaMgO_full_tmp_1961 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1962 CaMgO_full_tmp_1962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1963 CaMgO_full_tmp_1963 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1964 CaMgO_full_tmp_1964 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1965 CaMgO_full_tmp_1965 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1966 CaMgO_full_tmp_1966 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1967 CaMgO_full_tmp_1967 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1968 CaMgO_full_tmp_1968 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1969 CaMgO_full_tmp_1969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1970 CaMgO_full_tmp_1970 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1971 CaMgO_full_tmp_1971 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1972 CaMgO_full_tmp_1972 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1973 CaMgO_full_tmp_1973 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1974 CaMgO_full_tmp_1974 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1975 CaMgO_full_tmp_1975 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1976 CaMgO_full_tmp_1976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1977 CaMgO_full_tmp_1977 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1978 CaMgO_full_tmp_1978 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1979 CaMgO_full_tmp_1979 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1980 CaMgO_full_tmp_1980 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1981 CaMgO_full_tmp_1981 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1982 CaMgO_full_tmp_1982 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1983 CaMgO_full_tmp_1983 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1984 CaMgO_full_tmp_1984 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1985 CaMgO_full_tmp_1985 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1986 CaMgO_full_tmp_1986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1987 CaMgO_full_tmp_1987 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1988 CaMgO_full_tmp_1988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1989 CaMgO_full_tmp_1989 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1990 CaMgO_full_tmp_1990 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1991 CaMgO_full_tmp_1991 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1992 CaMgO_full_tmp_1992 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1993 CaMgO_full_tmp_1993 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1994 CaMgO_full_tmp_1994 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1995 CaMgO_full_tmp_1995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1996 CaMgO_full_tmp_1996 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1997 CaMgO_full_tmp_1997 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1998 CaMgO_full_tmp_1998 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1999 CaMgO_full_tmp_1999 &

wait
mv CaMgO_full_tmp_1900.f9 CaMgO_full_1900.f9
mv CaMgO_full_tmp_1900.gui CaMgO_full_1900.gui
mv CaMgO_full_tmp_1901.f9 CaMgO_full_1901.f9
mv CaMgO_full_tmp_1901.gui CaMgO_full_1901.gui
mv CaMgO_full_tmp_1902.f9 CaMgO_full_1902.f9
mv CaMgO_full_tmp_1902.gui CaMgO_full_1902.gui
mv CaMgO_full_tmp_1903.f9 CaMgO_full_1903.f9
mv CaMgO_full_tmp_1903.gui CaMgO_full_1903.gui
mv CaMgO_full_tmp_1904.f9 CaMgO_full_1904.f9
mv CaMgO_full_tmp_1904.gui CaMgO_full_1904.gui
mv CaMgO_full_tmp_1905.f9 CaMgO_full_1905.f9
mv CaMgO_full_tmp_1905.gui CaMgO_full_1905.gui
mv CaMgO_full_tmp_1906.f9 CaMgO_full_1906.f9
mv CaMgO_full_tmp_1906.gui CaMgO_full_1906.gui
mv CaMgO_full_tmp_1907.f9 CaMgO_full_1907.f9
mv CaMgO_full_tmp_1907.gui CaMgO_full_1907.gui
mv CaMgO_full_tmp_1908.f9 CaMgO_full_1908.f9
mv CaMgO_full_tmp_1908.gui CaMgO_full_1908.gui
mv CaMgO_full_tmp_1909.f9 CaMgO_full_1909.f9
mv CaMgO_full_tmp_1909.gui CaMgO_full_1909.gui
mv CaMgO_full_tmp_1910.f9 CaMgO_full_1910.f9
mv CaMgO_full_tmp_1910.gui CaMgO_full_1910.gui
mv CaMgO_full_tmp_1911.f9 CaMgO_full_1911.f9
mv CaMgO_full_tmp_1911.gui CaMgO_full_1911.gui
mv CaMgO_full_tmp_1912.f9 CaMgO_full_1912.f9
mv CaMgO_full_tmp_1912.gui CaMgO_full_1912.gui
mv CaMgO_full_tmp_1913.f9 CaMgO_full_1913.f9
mv CaMgO_full_tmp_1913.gui CaMgO_full_1913.gui
mv CaMgO_full_tmp_1914.f9 CaMgO_full_1914.f9
mv CaMgO_full_tmp_1914.gui CaMgO_full_1914.gui
mv CaMgO_full_tmp_1915.f9 CaMgO_full_1915.f9
mv CaMgO_full_tmp_1915.gui CaMgO_full_1915.gui
mv CaMgO_full_tmp_1916.f9 CaMgO_full_1916.f9
mv CaMgO_full_tmp_1916.gui CaMgO_full_1916.gui
mv CaMgO_full_tmp_1917.f9 CaMgO_full_1917.f9
mv CaMgO_full_tmp_1917.gui CaMgO_full_1917.gui
mv CaMgO_full_tmp_1918.f9 CaMgO_full_1918.f9
mv CaMgO_full_tmp_1918.gui CaMgO_full_1918.gui
mv CaMgO_full_tmp_1919.f9 CaMgO_full_1919.f9
mv CaMgO_full_tmp_1919.gui CaMgO_full_1919.gui
mv CaMgO_full_tmp_1920.f9 CaMgO_full_1920.f9
mv CaMgO_full_tmp_1920.gui CaMgO_full_1920.gui
mv CaMgO_full_tmp_1921.f9 CaMgO_full_1921.f9
mv CaMgO_full_tmp_1921.gui CaMgO_full_1921.gui
mv CaMgO_full_tmp_1922.f9 CaMgO_full_1922.f9
mv CaMgO_full_tmp_1922.gui CaMgO_full_1922.gui
mv CaMgO_full_tmp_1923.f9 CaMgO_full_1923.f9
mv CaMgO_full_tmp_1923.gui CaMgO_full_1923.gui
mv CaMgO_full_tmp_1924.f9 CaMgO_full_1924.f9
mv CaMgO_full_tmp_1924.gui CaMgO_full_1924.gui
mv CaMgO_full_tmp_1925.f9 CaMgO_full_1925.f9
mv CaMgO_full_tmp_1925.gui CaMgO_full_1925.gui
mv CaMgO_full_tmp_1926.f9 CaMgO_full_1926.f9
mv CaMgO_full_tmp_1926.gui CaMgO_full_1926.gui
mv CaMgO_full_tmp_1927.f9 CaMgO_full_1927.f9
mv CaMgO_full_tmp_1927.gui CaMgO_full_1927.gui
mv CaMgO_full_tmp_1928.f9 CaMgO_full_1928.f9
mv CaMgO_full_tmp_1928.gui CaMgO_full_1928.gui
mv CaMgO_full_tmp_1929.f9 CaMgO_full_1929.f9
mv CaMgO_full_tmp_1929.gui CaMgO_full_1929.gui
mv CaMgO_full_tmp_1930.f9 CaMgO_full_1930.f9
mv CaMgO_full_tmp_1930.gui CaMgO_full_1930.gui
mv CaMgO_full_tmp_1931.f9 CaMgO_full_1931.f9
mv CaMgO_full_tmp_1931.gui CaMgO_full_1931.gui
mv CaMgO_full_tmp_1932.f9 CaMgO_full_1932.f9
mv CaMgO_full_tmp_1932.gui CaMgO_full_1932.gui
mv CaMgO_full_tmp_1933.f9 CaMgO_full_1933.f9
mv CaMgO_full_tmp_1933.gui CaMgO_full_1933.gui
mv CaMgO_full_tmp_1934.f9 CaMgO_full_1934.f9
mv CaMgO_full_tmp_1934.gui CaMgO_full_1934.gui
mv CaMgO_full_tmp_1935.f9 CaMgO_full_1935.f9
mv CaMgO_full_tmp_1935.gui CaMgO_full_1935.gui
mv CaMgO_full_tmp_1936.f9 CaMgO_full_1936.f9
mv CaMgO_full_tmp_1936.gui CaMgO_full_1936.gui
mv CaMgO_full_tmp_1937.f9 CaMgO_full_1937.f9
mv CaMgO_full_tmp_1937.gui CaMgO_full_1937.gui
mv CaMgO_full_tmp_1938.f9 CaMgO_full_1938.f9
mv CaMgO_full_tmp_1938.gui CaMgO_full_1938.gui
mv CaMgO_full_tmp_1939.f9 CaMgO_full_1939.f9
mv CaMgO_full_tmp_1939.gui CaMgO_full_1939.gui
mv CaMgO_full_tmp_1940.f9 CaMgO_full_1940.f9
mv CaMgO_full_tmp_1940.gui CaMgO_full_1940.gui
mv CaMgO_full_tmp_1941.f9 CaMgO_full_1941.f9
mv CaMgO_full_tmp_1941.gui CaMgO_full_1941.gui
mv CaMgO_full_tmp_1942.f9 CaMgO_full_1942.f9
mv CaMgO_full_tmp_1942.gui CaMgO_full_1942.gui
mv CaMgO_full_tmp_1943.f9 CaMgO_full_1943.f9
mv CaMgO_full_tmp_1943.gui CaMgO_full_1943.gui
mv CaMgO_full_tmp_1944.f9 CaMgO_full_1944.f9
mv CaMgO_full_tmp_1944.gui CaMgO_full_1944.gui
mv CaMgO_full_tmp_1945.f9 CaMgO_full_1945.f9
mv CaMgO_full_tmp_1945.gui CaMgO_full_1945.gui
mv CaMgO_full_tmp_1946.f9 CaMgO_full_1946.f9
mv CaMgO_full_tmp_1946.gui CaMgO_full_1946.gui
mv CaMgO_full_tmp_1947.f9 CaMgO_full_1947.f9
mv CaMgO_full_tmp_1947.gui CaMgO_full_1947.gui
mv CaMgO_full_tmp_1948.f9 CaMgO_full_1948.f9
mv CaMgO_full_tmp_1948.gui CaMgO_full_1948.gui
mv CaMgO_full_tmp_1949.f9 CaMgO_full_1949.f9
mv CaMgO_full_tmp_1949.gui CaMgO_full_1949.gui
mv CaMgO_full_tmp_1950.f9 CaMgO_full_1950.f9
mv CaMgO_full_tmp_1950.gui CaMgO_full_1950.gui
mv CaMgO_full_tmp_1951.f9 CaMgO_full_1951.f9
mv CaMgO_full_tmp_1951.gui CaMgO_full_1951.gui
mv CaMgO_full_tmp_1952.f9 CaMgO_full_1952.f9
mv CaMgO_full_tmp_1952.gui CaMgO_full_1952.gui
mv CaMgO_full_tmp_1953.f9 CaMgO_full_1953.f9
mv CaMgO_full_tmp_1953.gui CaMgO_full_1953.gui
mv CaMgO_full_tmp_1954.f9 CaMgO_full_1954.f9
mv CaMgO_full_tmp_1954.gui CaMgO_full_1954.gui
mv CaMgO_full_tmp_1955.f9 CaMgO_full_1955.f9
mv CaMgO_full_tmp_1955.gui CaMgO_full_1955.gui
mv CaMgO_full_tmp_1956.f9 CaMgO_full_1956.f9
mv CaMgO_full_tmp_1956.gui CaMgO_full_1956.gui
mv CaMgO_full_tmp_1957.f9 CaMgO_full_1957.f9
mv CaMgO_full_tmp_1957.gui CaMgO_full_1957.gui
mv CaMgO_full_tmp_1958.f9 CaMgO_full_1958.f9
mv CaMgO_full_tmp_1958.gui CaMgO_full_1958.gui
mv CaMgO_full_tmp_1959.f9 CaMgO_full_1959.f9
mv CaMgO_full_tmp_1959.gui CaMgO_full_1959.gui
mv CaMgO_full_tmp_1960.f9 CaMgO_full_1960.f9
mv CaMgO_full_tmp_1960.gui CaMgO_full_1960.gui
mv CaMgO_full_tmp_1961.f9 CaMgO_full_1961.f9
mv CaMgO_full_tmp_1961.gui CaMgO_full_1961.gui
mv CaMgO_full_tmp_1962.f9 CaMgO_full_1962.f9
mv CaMgO_full_tmp_1962.gui CaMgO_full_1962.gui
mv CaMgO_full_tmp_1963.f9 CaMgO_full_1963.f9
mv CaMgO_full_tmp_1963.gui CaMgO_full_1963.gui
mv CaMgO_full_tmp_1964.f9 CaMgO_full_1964.f9
mv CaMgO_full_tmp_1964.gui CaMgO_full_1964.gui
mv CaMgO_full_tmp_1965.f9 CaMgO_full_1965.f9
mv CaMgO_full_tmp_1965.gui CaMgO_full_1965.gui
mv CaMgO_full_tmp_1966.f9 CaMgO_full_1966.f9
mv CaMgO_full_tmp_1966.gui CaMgO_full_1966.gui
mv CaMgO_full_tmp_1967.f9 CaMgO_full_1967.f9
mv CaMgO_full_tmp_1967.gui CaMgO_full_1967.gui
mv CaMgO_full_tmp_1968.f9 CaMgO_full_1968.f9
mv CaMgO_full_tmp_1968.gui CaMgO_full_1968.gui
mv CaMgO_full_tmp_1969.f9 CaMgO_full_1969.f9
mv CaMgO_full_tmp_1969.gui CaMgO_full_1969.gui
mv CaMgO_full_tmp_1970.f9 CaMgO_full_1970.f9
mv CaMgO_full_tmp_1970.gui CaMgO_full_1970.gui
mv CaMgO_full_tmp_1971.f9 CaMgO_full_1971.f9
mv CaMgO_full_tmp_1971.gui CaMgO_full_1971.gui
mv CaMgO_full_tmp_1972.f9 CaMgO_full_1972.f9
mv CaMgO_full_tmp_1972.gui CaMgO_full_1972.gui
mv CaMgO_full_tmp_1973.f9 CaMgO_full_1973.f9
mv CaMgO_full_tmp_1973.gui CaMgO_full_1973.gui
mv CaMgO_full_tmp_1974.f9 CaMgO_full_1974.f9
mv CaMgO_full_tmp_1974.gui CaMgO_full_1974.gui
mv CaMgO_full_tmp_1975.f9 CaMgO_full_1975.f9
mv CaMgO_full_tmp_1975.gui CaMgO_full_1975.gui
mv CaMgO_full_tmp_1976.f9 CaMgO_full_1976.f9
mv CaMgO_full_tmp_1976.gui CaMgO_full_1976.gui
mv CaMgO_full_tmp_1977.f9 CaMgO_full_1977.f9
mv CaMgO_full_tmp_1977.gui CaMgO_full_1977.gui
mv CaMgO_full_tmp_1978.f9 CaMgO_full_1978.f9
mv CaMgO_full_tmp_1978.gui CaMgO_full_1978.gui
mv CaMgO_full_tmp_1979.f9 CaMgO_full_1979.f9
mv CaMgO_full_tmp_1979.gui CaMgO_full_1979.gui
mv CaMgO_full_tmp_1980.f9 CaMgO_full_1980.f9
mv CaMgO_full_tmp_1980.gui CaMgO_full_1980.gui
mv CaMgO_full_tmp_1981.f9 CaMgO_full_1981.f9
mv CaMgO_full_tmp_1981.gui CaMgO_full_1981.gui
mv CaMgO_full_tmp_1982.f9 CaMgO_full_1982.f9
mv CaMgO_full_tmp_1982.gui CaMgO_full_1982.gui
mv CaMgO_full_tmp_1983.f9 CaMgO_full_1983.f9
mv CaMgO_full_tmp_1983.gui CaMgO_full_1983.gui
mv CaMgO_full_tmp_1984.f9 CaMgO_full_1984.f9
mv CaMgO_full_tmp_1984.gui CaMgO_full_1984.gui
mv CaMgO_full_tmp_1985.f9 CaMgO_full_1985.f9
mv CaMgO_full_tmp_1985.gui CaMgO_full_1985.gui
mv CaMgO_full_tmp_1986.f9 CaMgO_full_1986.f9
mv CaMgO_full_tmp_1986.gui CaMgO_full_1986.gui
mv CaMgO_full_tmp_1987.f9 CaMgO_full_1987.f9
mv CaMgO_full_tmp_1987.gui CaMgO_full_1987.gui
mv CaMgO_full_tmp_1988.f9 CaMgO_full_1988.f9
mv CaMgO_full_tmp_1988.gui CaMgO_full_1988.gui
mv CaMgO_full_tmp_1989.f9 CaMgO_full_1989.f9
mv CaMgO_full_tmp_1989.gui CaMgO_full_1989.gui
mv CaMgO_full_tmp_1990.f9 CaMgO_full_1990.f9
mv CaMgO_full_tmp_1990.gui CaMgO_full_1990.gui
mv CaMgO_full_tmp_1991.f9 CaMgO_full_1991.f9
mv CaMgO_full_tmp_1991.gui CaMgO_full_1991.gui
mv CaMgO_full_tmp_1992.f9 CaMgO_full_1992.f9
mv CaMgO_full_tmp_1992.gui CaMgO_full_1992.gui
mv CaMgO_full_tmp_1993.f9 CaMgO_full_1993.f9
mv CaMgO_full_tmp_1993.gui CaMgO_full_1993.gui
mv CaMgO_full_tmp_1994.f9 CaMgO_full_1994.f9
mv CaMgO_full_tmp_1994.gui CaMgO_full_1994.gui
mv CaMgO_full_tmp_1995.f9 CaMgO_full_1995.f9
mv CaMgO_full_tmp_1995.gui CaMgO_full_1995.gui
mv CaMgO_full_tmp_1996.f9 CaMgO_full_1996.f9
mv CaMgO_full_tmp_1996.gui CaMgO_full_1996.gui
mv CaMgO_full_tmp_1997.f9 CaMgO_full_1997.f9
mv CaMgO_full_tmp_1997.gui CaMgO_full_1997.gui
mv CaMgO_full_tmp_1998.f9 CaMgO_full_1998.f9
mv CaMgO_full_tmp_1998.gui CaMgO_full_1998.gui
mv CaMgO_full_tmp_1999.f9 CaMgO_full_1999.f9
mv CaMgO_full_tmp_1999.gui CaMgO_full_1999.gui
