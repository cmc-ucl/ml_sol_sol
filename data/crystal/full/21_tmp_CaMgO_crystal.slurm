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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2000 CaMgO_full_tmp_2000 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2001 CaMgO_full_tmp_2001 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2002 CaMgO_full_tmp_2002 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2003 CaMgO_full_tmp_2003 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2004 CaMgO_full_tmp_2004 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2005 CaMgO_full_tmp_2005 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2006 CaMgO_full_tmp_2006 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2007 CaMgO_full_tmp_2007 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2008 CaMgO_full_tmp_2008 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2009 CaMgO_full_tmp_2009 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2010 CaMgO_full_tmp_2010 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2011 CaMgO_full_tmp_2011 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2012 CaMgO_full_tmp_2012 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2013 CaMgO_full_tmp_2013 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2014 CaMgO_full_tmp_2014 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2015 CaMgO_full_tmp_2015 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2016 CaMgO_full_tmp_2016 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2017 CaMgO_full_tmp_2017 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2018 CaMgO_full_tmp_2018 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2019 CaMgO_full_tmp_2019 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2020 CaMgO_full_tmp_2020 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2021 CaMgO_full_tmp_2021 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2022 CaMgO_full_tmp_2022 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2023 CaMgO_full_tmp_2023 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2024 CaMgO_full_tmp_2024 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2025 CaMgO_full_tmp_2025 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2026 CaMgO_full_tmp_2026 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2027 CaMgO_full_tmp_2027 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2028 CaMgO_full_tmp_2028 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2029 CaMgO_full_tmp_2029 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2030 CaMgO_full_tmp_2030 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2031 CaMgO_full_tmp_2031 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2032 CaMgO_full_tmp_2032 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2033 CaMgO_full_tmp_2033 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2034 CaMgO_full_tmp_2034 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2035 CaMgO_full_tmp_2035 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2036 CaMgO_full_tmp_2036 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2037 CaMgO_full_tmp_2037 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2038 CaMgO_full_tmp_2038 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2039 CaMgO_full_tmp_2039 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2040 CaMgO_full_tmp_2040 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2041 CaMgO_full_tmp_2041 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2042 CaMgO_full_tmp_2042 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2043 CaMgO_full_tmp_2043 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2044 CaMgO_full_tmp_2044 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2045 CaMgO_full_tmp_2045 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2046 CaMgO_full_tmp_2046 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2047 CaMgO_full_tmp_2047 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2048 CaMgO_full_tmp_2048 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2049 CaMgO_full_tmp_2049 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2050 CaMgO_full_tmp_2050 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2051 CaMgO_full_tmp_2051 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2052 CaMgO_full_tmp_2052 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2053 CaMgO_full_tmp_2053 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2054 CaMgO_full_tmp_2054 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2055 CaMgO_full_tmp_2055 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2056 CaMgO_full_tmp_2056 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2057 CaMgO_full_tmp_2057 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2058 CaMgO_full_tmp_2058 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2059 CaMgO_full_tmp_2059 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2060 CaMgO_full_tmp_2060 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2061 CaMgO_full_tmp_2061 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2062 CaMgO_full_tmp_2062 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2063 CaMgO_full_tmp_2063 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2064 CaMgO_full_tmp_2064 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2065 CaMgO_full_tmp_2065 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2066 CaMgO_full_tmp_2066 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2067 CaMgO_full_tmp_2067 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2068 CaMgO_full_tmp_2068 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2069 CaMgO_full_tmp_2069 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2070 CaMgO_full_tmp_2070 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2071 CaMgO_full_tmp_2071 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2072 CaMgO_full_tmp_2072 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2073 CaMgO_full_tmp_2073 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2074 CaMgO_full_tmp_2074 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2075 CaMgO_full_tmp_2075 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2076 CaMgO_full_tmp_2076 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2077 CaMgO_full_tmp_2077 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2078 CaMgO_full_tmp_2078 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2079 CaMgO_full_tmp_2079 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2080 CaMgO_full_tmp_2080 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2081 CaMgO_full_tmp_2081 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2082 CaMgO_full_tmp_2082 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2083 CaMgO_full_tmp_2083 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2084 CaMgO_full_tmp_2084 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2085 CaMgO_full_tmp_2085 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2086 CaMgO_full_tmp_2086 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2087 CaMgO_full_tmp_2087 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2088 CaMgO_full_tmp_2088 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2089 CaMgO_full_tmp_2089 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2090 CaMgO_full_tmp_2090 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2091 CaMgO_full_tmp_2091 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2092 CaMgO_full_tmp_2092 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2093 CaMgO_full_tmp_2093 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2094 CaMgO_full_tmp_2094 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2095 CaMgO_full_tmp_2095 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2096 CaMgO_full_tmp_2096 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2097 CaMgO_full_tmp_2097 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2098 CaMgO_full_tmp_2098 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2099 CaMgO_full_tmp_2099 &

wait
mv CaMgO_full_tmp_2000.f9 CaMgO_full_2000.f9
mv CaMgO_full_tmp_2000.gui CaMgO_full_2000.gui
mv CaMgO_full_tmp_2001.f9 CaMgO_full_2001.f9
mv CaMgO_full_tmp_2001.gui CaMgO_full_2001.gui
mv CaMgO_full_tmp_2002.f9 CaMgO_full_2002.f9
mv CaMgO_full_tmp_2002.gui CaMgO_full_2002.gui
mv CaMgO_full_tmp_2003.f9 CaMgO_full_2003.f9
mv CaMgO_full_tmp_2003.gui CaMgO_full_2003.gui
mv CaMgO_full_tmp_2004.f9 CaMgO_full_2004.f9
mv CaMgO_full_tmp_2004.gui CaMgO_full_2004.gui
mv CaMgO_full_tmp_2005.f9 CaMgO_full_2005.f9
mv CaMgO_full_tmp_2005.gui CaMgO_full_2005.gui
mv CaMgO_full_tmp_2006.f9 CaMgO_full_2006.f9
mv CaMgO_full_tmp_2006.gui CaMgO_full_2006.gui
mv CaMgO_full_tmp_2007.f9 CaMgO_full_2007.f9
mv CaMgO_full_tmp_2007.gui CaMgO_full_2007.gui
mv CaMgO_full_tmp_2008.f9 CaMgO_full_2008.f9
mv CaMgO_full_tmp_2008.gui CaMgO_full_2008.gui
mv CaMgO_full_tmp_2009.f9 CaMgO_full_2009.f9
mv CaMgO_full_tmp_2009.gui CaMgO_full_2009.gui
mv CaMgO_full_tmp_2010.f9 CaMgO_full_2010.f9
mv CaMgO_full_tmp_2010.gui CaMgO_full_2010.gui
mv CaMgO_full_tmp_2011.f9 CaMgO_full_2011.f9
mv CaMgO_full_tmp_2011.gui CaMgO_full_2011.gui
mv CaMgO_full_tmp_2012.f9 CaMgO_full_2012.f9
mv CaMgO_full_tmp_2012.gui CaMgO_full_2012.gui
mv CaMgO_full_tmp_2013.f9 CaMgO_full_2013.f9
mv CaMgO_full_tmp_2013.gui CaMgO_full_2013.gui
mv CaMgO_full_tmp_2014.f9 CaMgO_full_2014.f9
mv CaMgO_full_tmp_2014.gui CaMgO_full_2014.gui
mv CaMgO_full_tmp_2015.f9 CaMgO_full_2015.f9
mv CaMgO_full_tmp_2015.gui CaMgO_full_2015.gui
mv CaMgO_full_tmp_2016.f9 CaMgO_full_2016.f9
mv CaMgO_full_tmp_2016.gui CaMgO_full_2016.gui
mv CaMgO_full_tmp_2017.f9 CaMgO_full_2017.f9
mv CaMgO_full_tmp_2017.gui CaMgO_full_2017.gui
mv CaMgO_full_tmp_2018.f9 CaMgO_full_2018.f9
mv CaMgO_full_tmp_2018.gui CaMgO_full_2018.gui
mv CaMgO_full_tmp_2019.f9 CaMgO_full_2019.f9
mv CaMgO_full_tmp_2019.gui CaMgO_full_2019.gui
mv CaMgO_full_tmp_2020.f9 CaMgO_full_2020.f9
mv CaMgO_full_tmp_2020.gui CaMgO_full_2020.gui
mv CaMgO_full_tmp_2021.f9 CaMgO_full_2021.f9
mv CaMgO_full_tmp_2021.gui CaMgO_full_2021.gui
mv CaMgO_full_tmp_2022.f9 CaMgO_full_2022.f9
mv CaMgO_full_tmp_2022.gui CaMgO_full_2022.gui
mv CaMgO_full_tmp_2023.f9 CaMgO_full_2023.f9
mv CaMgO_full_tmp_2023.gui CaMgO_full_2023.gui
mv CaMgO_full_tmp_2024.f9 CaMgO_full_2024.f9
mv CaMgO_full_tmp_2024.gui CaMgO_full_2024.gui
mv CaMgO_full_tmp_2025.f9 CaMgO_full_2025.f9
mv CaMgO_full_tmp_2025.gui CaMgO_full_2025.gui
mv CaMgO_full_tmp_2026.f9 CaMgO_full_2026.f9
mv CaMgO_full_tmp_2026.gui CaMgO_full_2026.gui
mv CaMgO_full_tmp_2027.f9 CaMgO_full_2027.f9
mv CaMgO_full_tmp_2027.gui CaMgO_full_2027.gui
mv CaMgO_full_tmp_2028.f9 CaMgO_full_2028.f9
mv CaMgO_full_tmp_2028.gui CaMgO_full_2028.gui
mv CaMgO_full_tmp_2029.f9 CaMgO_full_2029.f9
mv CaMgO_full_tmp_2029.gui CaMgO_full_2029.gui
mv CaMgO_full_tmp_2030.f9 CaMgO_full_2030.f9
mv CaMgO_full_tmp_2030.gui CaMgO_full_2030.gui
mv CaMgO_full_tmp_2031.f9 CaMgO_full_2031.f9
mv CaMgO_full_tmp_2031.gui CaMgO_full_2031.gui
mv CaMgO_full_tmp_2032.f9 CaMgO_full_2032.f9
mv CaMgO_full_tmp_2032.gui CaMgO_full_2032.gui
mv CaMgO_full_tmp_2033.f9 CaMgO_full_2033.f9
mv CaMgO_full_tmp_2033.gui CaMgO_full_2033.gui
mv CaMgO_full_tmp_2034.f9 CaMgO_full_2034.f9
mv CaMgO_full_tmp_2034.gui CaMgO_full_2034.gui
mv CaMgO_full_tmp_2035.f9 CaMgO_full_2035.f9
mv CaMgO_full_tmp_2035.gui CaMgO_full_2035.gui
mv CaMgO_full_tmp_2036.f9 CaMgO_full_2036.f9
mv CaMgO_full_tmp_2036.gui CaMgO_full_2036.gui
mv CaMgO_full_tmp_2037.f9 CaMgO_full_2037.f9
mv CaMgO_full_tmp_2037.gui CaMgO_full_2037.gui
mv CaMgO_full_tmp_2038.f9 CaMgO_full_2038.f9
mv CaMgO_full_tmp_2038.gui CaMgO_full_2038.gui
mv CaMgO_full_tmp_2039.f9 CaMgO_full_2039.f9
mv CaMgO_full_tmp_2039.gui CaMgO_full_2039.gui
mv CaMgO_full_tmp_2040.f9 CaMgO_full_2040.f9
mv CaMgO_full_tmp_2040.gui CaMgO_full_2040.gui
mv CaMgO_full_tmp_2041.f9 CaMgO_full_2041.f9
mv CaMgO_full_tmp_2041.gui CaMgO_full_2041.gui
mv CaMgO_full_tmp_2042.f9 CaMgO_full_2042.f9
mv CaMgO_full_tmp_2042.gui CaMgO_full_2042.gui
mv CaMgO_full_tmp_2043.f9 CaMgO_full_2043.f9
mv CaMgO_full_tmp_2043.gui CaMgO_full_2043.gui
mv CaMgO_full_tmp_2044.f9 CaMgO_full_2044.f9
mv CaMgO_full_tmp_2044.gui CaMgO_full_2044.gui
mv CaMgO_full_tmp_2045.f9 CaMgO_full_2045.f9
mv CaMgO_full_tmp_2045.gui CaMgO_full_2045.gui
mv CaMgO_full_tmp_2046.f9 CaMgO_full_2046.f9
mv CaMgO_full_tmp_2046.gui CaMgO_full_2046.gui
mv CaMgO_full_tmp_2047.f9 CaMgO_full_2047.f9
mv CaMgO_full_tmp_2047.gui CaMgO_full_2047.gui
mv CaMgO_full_tmp_2048.f9 CaMgO_full_2048.f9
mv CaMgO_full_tmp_2048.gui CaMgO_full_2048.gui
mv CaMgO_full_tmp_2049.f9 CaMgO_full_2049.f9
mv CaMgO_full_tmp_2049.gui CaMgO_full_2049.gui
mv CaMgO_full_tmp_2050.f9 CaMgO_full_2050.f9
mv CaMgO_full_tmp_2050.gui CaMgO_full_2050.gui
mv CaMgO_full_tmp_2051.f9 CaMgO_full_2051.f9
mv CaMgO_full_tmp_2051.gui CaMgO_full_2051.gui
mv CaMgO_full_tmp_2052.f9 CaMgO_full_2052.f9
mv CaMgO_full_tmp_2052.gui CaMgO_full_2052.gui
mv CaMgO_full_tmp_2053.f9 CaMgO_full_2053.f9
mv CaMgO_full_tmp_2053.gui CaMgO_full_2053.gui
mv CaMgO_full_tmp_2054.f9 CaMgO_full_2054.f9
mv CaMgO_full_tmp_2054.gui CaMgO_full_2054.gui
mv CaMgO_full_tmp_2055.f9 CaMgO_full_2055.f9
mv CaMgO_full_tmp_2055.gui CaMgO_full_2055.gui
mv CaMgO_full_tmp_2056.f9 CaMgO_full_2056.f9
mv CaMgO_full_tmp_2056.gui CaMgO_full_2056.gui
mv CaMgO_full_tmp_2057.f9 CaMgO_full_2057.f9
mv CaMgO_full_tmp_2057.gui CaMgO_full_2057.gui
mv CaMgO_full_tmp_2058.f9 CaMgO_full_2058.f9
mv CaMgO_full_tmp_2058.gui CaMgO_full_2058.gui
mv CaMgO_full_tmp_2059.f9 CaMgO_full_2059.f9
mv CaMgO_full_tmp_2059.gui CaMgO_full_2059.gui
mv CaMgO_full_tmp_2060.f9 CaMgO_full_2060.f9
mv CaMgO_full_tmp_2060.gui CaMgO_full_2060.gui
mv CaMgO_full_tmp_2061.f9 CaMgO_full_2061.f9
mv CaMgO_full_tmp_2061.gui CaMgO_full_2061.gui
mv CaMgO_full_tmp_2062.f9 CaMgO_full_2062.f9
mv CaMgO_full_tmp_2062.gui CaMgO_full_2062.gui
mv CaMgO_full_tmp_2063.f9 CaMgO_full_2063.f9
mv CaMgO_full_tmp_2063.gui CaMgO_full_2063.gui
mv CaMgO_full_tmp_2064.f9 CaMgO_full_2064.f9
mv CaMgO_full_tmp_2064.gui CaMgO_full_2064.gui
mv CaMgO_full_tmp_2065.f9 CaMgO_full_2065.f9
mv CaMgO_full_tmp_2065.gui CaMgO_full_2065.gui
mv CaMgO_full_tmp_2066.f9 CaMgO_full_2066.f9
mv CaMgO_full_tmp_2066.gui CaMgO_full_2066.gui
mv CaMgO_full_tmp_2067.f9 CaMgO_full_2067.f9
mv CaMgO_full_tmp_2067.gui CaMgO_full_2067.gui
mv CaMgO_full_tmp_2068.f9 CaMgO_full_2068.f9
mv CaMgO_full_tmp_2068.gui CaMgO_full_2068.gui
mv CaMgO_full_tmp_2069.f9 CaMgO_full_2069.f9
mv CaMgO_full_tmp_2069.gui CaMgO_full_2069.gui
mv CaMgO_full_tmp_2070.f9 CaMgO_full_2070.f9
mv CaMgO_full_tmp_2070.gui CaMgO_full_2070.gui
mv CaMgO_full_tmp_2071.f9 CaMgO_full_2071.f9
mv CaMgO_full_tmp_2071.gui CaMgO_full_2071.gui
mv CaMgO_full_tmp_2072.f9 CaMgO_full_2072.f9
mv CaMgO_full_tmp_2072.gui CaMgO_full_2072.gui
mv CaMgO_full_tmp_2073.f9 CaMgO_full_2073.f9
mv CaMgO_full_tmp_2073.gui CaMgO_full_2073.gui
mv CaMgO_full_tmp_2074.f9 CaMgO_full_2074.f9
mv CaMgO_full_tmp_2074.gui CaMgO_full_2074.gui
mv CaMgO_full_tmp_2075.f9 CaMgO_full_2075.f9
mv CaMgO_full_tmp_2075.gui CaMgO_full_2075.gui
mv CaMgO_full_tmp_2076.f9 CaMgO_full_2076.f9
mv CaMgO_full_tmp_2076.gui CaMgO_full_2076.gui
mv CaMgO_full_tmp_2077.f9 CaMgO_full_2077.f9
mv CaMgO_full_tmp_2077.gui CaMgO_full_2077.gui
mv CaMgO_full_tmp_2078.f9 CaMgO_full_2078.f9
mv CaMgO_full_tmp_2078.gui CaMgO_full_2078.gui
mv CaMgO_full_tmp_2079.f9 CaMgO_full_2079.f9
mv CaMgO_full_tmp_2079.gui CaMgO_full_2079.gui
mv CaMgO_full_tmp_2080.f9 CaMgO_full_2080.f9
mv CaMgO_full_tmp_2080.gui CaMgO_full_2080.gui
mv CaMgO_full_tmp_2081.f9 CaMgO_full_2081.f9
mv CaMgO_full_tmp_2081.gui CaMgO_full_2081.gui
mv CaMgO_full_tmp_2082.f9 CaMgO_full_2082.f9
mv CaMgO_full_tmp_2082.gui CaMgO_full_2082.gui
mv CaMgO_full_tmp_2083.f9 CaMgO_full_2083.f9
mv CaMgO_full_tmp_2083.gui CaMgO_full_2083.gui
mv CaMgO_full_tmp_2084.f9 CaMgO_full_2084.f9
mv CaMgO_full_tmp_2084.gui CaMgO_full_2084.gui
mv CaMgO_full_tmp_2085.f9 CaMgO_full_2085.f9
mv CaMgO_full_tmp_2085.gui CaMgO_full_2085.gui
mv CaMgO_full_tmp_2086.f9 CaMgO_full_2086.f9
mv CaMgO_full_tmp_2086.gui CaMgO_full_2086.gui
mv CaMgO_full_tmp_2087.f9 CaMgO_full_2087.f9
mv CaMgO_full_tmp_2087.gui CaMgO_full_2087.gui
mv CaMgO_full_tmp_2088.f9 CaMgO_full_2088.f9
mv CaMgO_full_tmp_2088.gui CaMgO_full_2088.gui
mv CaMgO_full_tmp_2089.f9 CaMgO_full_2089.f9
mv CaMgO_full_tmp_2089.gui CaMgO_full_2089.gui
mv CaMgO_full_tmp_2090.f9 CaMgO_full_2090.f9
mv CaMgO_full_tmp_2090.gui CaMgO_full_2090.gui
mv CaMgO_full_tmp_2091.f9 CaMgO_full_2091.f9
mv CaMgO_full_tmp_2091.gui CaMgO_full_2091.gui
mv CaMgO_full_tmp_2092.f9 CaMgO_full_2092.f9
mv CaMgO_full_tmp_2092.gui CaMgO_full_2092.gui
mv CaMgO_full_tmp_2093.f9 CaMgO_full_2093.f9
mv CaMgO_full_tmp_2093.gui CaMgO_full_2093.gui
mv CaMgO_full_tmp_2094.f9 CaMgO_full_2094.f9
mv CaMgO_full_tmp_2094.gui CaMgO_full_2094.gui
mv CaMgO_full_tmp_2095.f9 CaMgO_full_2095.f9
mv CaMgO_full_tmp_2095.gui CaMgO_full_2095.gui
mv CaMgO_full_tmp_2096.f9 CaMgO_full_2096.f9
mv CaMgO_full_tmp_2096.gui CaMgO_full_2096.gui
mv CaMgO_full_tmp_2097.f9 CaMgO_full_2097.f9
mv CaMgO_full_tmp_2097.gui CaMgO_full_2097.gui
mv CaMgO_full_tmp_2098.f9 CaMgO_full_2098.f9
mv CaMgO_full_tmp_2098.gui CaMgO_full_2098.gui
mv CaMgO_full_tmp_2099.f9 CaMgO_full_2099.f9
mv CaMgO_full_tmp_2099.gui CaMgO_full_2099.gui
