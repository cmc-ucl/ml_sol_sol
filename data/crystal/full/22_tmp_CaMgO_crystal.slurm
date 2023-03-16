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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2100 CaMgO_full_tmp_2100 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2101 CaMgO_full_tmp_2101 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2102 CaMgO_full_tmp_2102 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2103 CaMgO_full_tmp_2103 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2104 CaMgO_full_tmp_2104 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2105 CaMgO_full_tmp_2105 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2106 CaMgO_full_tmp_2106 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2107 CaMgO_full_tmp_2107 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2108 CaMgO_full_tmp_2108 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2109 CaMgO_full_tmp_2109 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2110 CaMgO_full_tmp_2110 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2111 CaMgO_full_tmp_2111 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2112 CaMgO_full_tmp_2112 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2113 CaMgO_full_tmp_2113 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2114 CaMgO_full_tmp_2114 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2115 CaMgO_full_tmp_2115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2116 CaMgO_full_tmp_2116 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2117 CaMgO_full_tmp_2117 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2118 CaMgO_full_tmp_2118 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2119 CaMgO_full_tmp_2119 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2120 CaMgO_full_tmp_2120 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2121 CaMgO_full_tmp_2121 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2122 CaMgO_full_tmp_2122 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2123 CaMgO_full_tmp_2123 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2124 CaMgO_full_tmp_2124 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2125 CaMgO_full_tmp_2125 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2126 CaMgO_full_tmp_2126 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2127 CaMgO_full_tmp_2127 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2128 CaMgO_full_tmp_2128 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2129 CaMgO_full_tmp_2129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2130 CaMgO_full_tmp_2130 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2131 CaMgO_full_tmp_2131 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2132 CaMgO_full_tmp_2132 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2133 CaMgO_full_tmp_2133 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2134 CaMgO_full_tmp_2134 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2135 CaMgO_full_tmp_2135 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2136 CaMgO_full_tmp_2136 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2137 CaMgO_full_tmp_2137 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2138 CaMgO_full_tmp_2138 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2139 CaMgO_full_tmp_2139 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2140 CaMgO_full_tmp_2140 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2141 CaMgO_full_tmp_2141 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2142 CaMgO_full_tmp_2142 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2143 CaMgO_full_tmp_2143 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2144 CaMgO_full_tmp_2144 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2145 CaMgO_full_tmp_2145 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2146 CaMgO_full_tmp_2146 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2147 CaMgO_full_tmp_2147 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2148 CaMgO_full_tmp_2148 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2149 CaMgO_full_tmp_2149 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2150 CaMgO_full_tmp_2150 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2151 CaMgO_full_tmp_2151 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2152 CaMgO_full_tmp_2152 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2153 CaMgO_full_tmp_2153 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2154 CaMgO_full_tmp_2154 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2155 CaMgO_full_tmp_2155 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2156 CaMgO_full_tmp_2156 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2157 CaMgO_full_tmp_2157 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2158 CaMgO_full_tmp_2158 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2159 CaMgO_full_tmp_2159 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2160 CaMgO_full_tmp_2160 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2161 CaMgO_full_tmp_2161 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2162 CaMgO_full_tmp_2162 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2163 CaMgO_full_tmp_2163 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2164 CaMgO_full_tmp_2164 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2165 CaMgO_full_tmp_2165 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2166 CaMgO_full_tmp_2166 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2167 CaMgO_full_tmp_2167 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2168 CaMgO_full_tmp_2168 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2169 CaMgO_full_tmp_2169 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2170 CaMgO_full_tmp_2170 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2171 CaMgO_full_tmp_2171 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2172 CaMgO_full_tmp_2172 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2173 CaMgO_full_tmp_2173 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2174 CaMgO_full_tmp_2174 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2175 CaMgO_full_tmp_2175 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2176 CaMgO_full_tmp_2176 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2177 CaMgO_full_tmp_2177 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2178 CaMgO_full_tmp_2178 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2179 CaMgO_full_tmp_2179 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2180 CaMgO_full_tmp_2180 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2181 CaMgO_full_tmp_2181 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2182 CaMgO_full_tmp_2182 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2183 CaMgO_full_tmp_2183 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2184 CaMgO_full_tmp_2184 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2185 CaMgO_full_tmp_2185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2186 CaMgO_full_tmp_2186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2187 CaMgO_full_tmp_2187 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2188 CaMgO_full_tmp_2188 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2189 CaMgO_full_tmp_2189 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2190 CaMgO_full_tmp_2190 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2191 CaMgO_full_tmp_2191 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2192 CaMgO_full_tmp_2192 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2193 CaMgO_full_tmp_2193 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2194 CaMgO_full_tmp_2194 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2195 CaMgO_full_tmp_2195 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2196 CaMgO_full_tmp_2196 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2197 CaMgO_full_tmp_2197 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2198 CaMgO_full_tmp_2198 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2199 CaMgO_full_tmp_2199 &

wait
mv CaMgO_full_tmp_2100.f9 CaMgO_full_2100.f9
mv CaMgO_full_tmp_2100.gui CaMgO_full_2100.gui
mv CaMgO_full_tmp_2101.f9 CaMgO_full_2101.f9
mv CaMgO_full_tmp_2101.gui CaMgO_full_2101.gui
mv CaMgO_full_tmp_2102.f9 CaMgO_full_2102.f9
mv CaMgO_full_tmp_2102.gui CaMgO_full_2102.gui
mv CaMgO_full_tmp_2103.f9 CaMgO_full_2103.f9
mv CaMgO_full_tmp_2103.gui CaMgO_full_2103.gui
mv CaMgO_full_tmp_2104.f9 CaMgO_full_2104.f9
mv CaMgO_full_tmp_2104.gui CaMgO_full_2104.gui
mv CaMgO_full_tmp_2105.f9 CaMgO_full_2105.f9
mv CaMgO_full_tmp_2105.gui CaMgO_full_2105.gui
mv CaMgO_full_tmp_2106.f9 CaMgO_full_2106.f9
mv CaMgO_full_tmp_2106.gui CaMgO_full_2106.gui
mv CaMgO_full_tmp_2107.f9 CaMgO_full_2107.f9
mv CaMgO_full_tmp_2107.gui CaMgO_full_2107.gui
mv CaMgO_full_tmp_2108.f9 CaMgO_full_2108.f9
mv CaMgO_full_tmp_2108.gui CaMgO_full_2108.gui
mv CaMgO_full_tmp_2109.f9 CaMgO_full_2109.f9
mv CaMgO_full_tmp_2109.gui CaMgO_full_2109.gui
mv CaMgO_full_tmp_2110.f9 CaMgO_full_2110.f9
mv CaMgO_full_tmp_2110.gui CaMgO_full_2110.gui
mv CaMgO_full_tmp_2111.f9 CaMgO_full_2111.f9
mv CaMgO_full_tmp_2111.gui CaMgO_full_2111.gui
mv CaMgO_full_tmp_2112.f9 CaMgO_full_2112.f9
mv CaMgO_full_tmp_2112.gui CaMgO_full_2112.gui
mv CaMgO_full_tmp_2113.f9 CaMgO_full_2113.f9
mv CaMgO_full_tmp_2113.gui CaMgO_full_2113.gui
mv CaMgO_full_tmp_2114.f9 CaMgO_full_2114.f9
mv CaMgO_full_tmp_2114.gui CaMgO_full_2114.gui
mv CaMgO_full_tmp_2115.f9 CaMgO_full_2115.f9
mv CaMgO_full_tmp_2115.gui CaMgO_full_2115.gui
mv CaMgO_full_tmp_2116.f9 CaMgO_full_2116.f9
mv CaMgO_full_tmp_2116.gui CaMgO_full_2116.gui
mv CaMgO_full_tmp_2117.f9 CaMgO_full_2117.f9
mv CaMgO_full_tmp_2117.gui CaMgO_full_2117.gui
mv CaMgO_full_tmp_2118.f9 CaMgO_full_2118.f9
mv CaMgO_full_tmp_2118.gui CaMgO_full_2118.gui
mv CaMgO_full_tmp_2119.f9 CaMgO_full_2119.f9
mv CaMgO_full_tmp_2119.gui CaMgO_full_2119.gui
mv CaMgO_full_tmp_2120.f9 CaMgO_full_2120.f9
mv CaMgO_full_tmp_2120.gui CaMgO_full_2120.gui
mv CaMgO_full_tmp_2121.f9 CaMgO_full_2121.f9
mv CaMgO_full_tmp_2121.gui CaMgO_full_2121.gui
mv CaMgO_full_tmp_2122.f9 CaMgO_full_2122.f9
mv CaMgO_full_tmp_2122.gui CaMgO_full_2122.gui
mv CaMgO_full_tmp_2123.f9 CaMgO_full_2123.f9
mv CaMgO_full_tmp_2123.gui CaMgO_full_2123.gui
mv CaMgO_full_tmp_2124.f9 CaMgO_full_2124.f9
mv CaMgO_full_tmp_2124.gui CaMgO_full_2124.gui
mv CaMgO_full_tmp_2125.f9 CaMgO_full_2125.f9
mv CaMgO_full_tmp_2125.gui CaMgO_full_2125.gui
mv CaMgO_full_tmp_2126.f9 CaMgO_full_2126.f9
mv CaMgO_full_tmp_2126.gui CaMgO_full_2126.gui
mv CaMgO_full_tmp_2127.f9 CaMgO_full_2127.f9
mv CaMgO_full_tmp_2127.gui CaMgO_full_2127.gui
mv CaMgO_full_tmp_2128.f9 CaMgO_full_2128.f9
mv CaMgO_full_tmp_2128.gui CaMgO_full_2128.gui
mv CaMgO_full_tmp_2129.f9 CaMgO_full_2129.f9
mv CaMgO_full_tmp_2129.gui CaMgO_full_2129.gui
mv CaMgO_full_tmp_2130.f9 CaMgO_full_2130.f9
mv CaMgO_full_tmp_2130.gui CaMgO_full_2130.gui
mv CaMgO_full_tmp_2131.f9 CaMgO_full_2131.f9
mv CaMgO_full_tmp_2131.gui CaMgO_full_2131.gui
mv CaMgO_full_tmp_2132.f9 CaMgO_full_2132.f9
mv CaMgO_full_tmp_2132.gui CaMgO_full_2132.gui
mv CaMgO_full_tmp_2133.f9 CaMgO_full_2133.f9
mv CaMgO_full_tmp_2133.gui CaMgO_full_2133.gui
mv CaMgO_full_tmp_2134.f9 CaMgO_full_2134.f9
mv CaMgO_full_tmp_2134.gui CaMgO_full_2134.gui
mv CaMgO_full_tmp_2135.f9 CaMgO_full_2135.f9
mv CaMgO_full_tmp_2135.gui CaMgO_full_2135.gui
mv CaMgO_full_tmp_2136.f9 CaMgO_full_2136.f9
mv CaMgO_full_tmp_2136.gui CaMgO_full_2136.gui
mv CaMgO_full_tmp_2137.f9 CaMgO_full_2137.f9
mv CaMgO_full_tmp_2137.gui CaMgO_full_2137.gui
mv CaMgO_full_tmp_2138.f9 CaMgO_full_2138.f9
mv CaMgO_full_tmp_2138.gui CaMgO_full_2138.gui
mv CaMgO_full_tmp_2139.f9 CaMgO_full_2139.f9
mv CaMgO_full_tmp_2139.gui CaMgO_full_2139.gui
mv CaMgO_full_tmp_2140.f9 CaMgO_full_2140.f9
mv CaMgO_full_tmp_2140.gui CaMgO_full_2140.gui
mv CaMgO_full_tmp_2141.f9 CaMgO_full_2141.f9
mv CaMgO_full_tmp_2141.gui CaMgO_full_2141.gui
mv CaMgO_full_tmp_2142.f9 CaMgO_full_2142.f9
mv CaMgO_full_tmp_2142.gui CaMgO_full_2142.gui
mv CaMgO_full_tmp_2143.f9 CaMgO_full_2143.f9
mv CaMgO_full_tmp_2143.gui CaMgO_full_2143.gui
mv CaMgO_full_tmp_2144.f9 CaMgO_full_2144.f9
mv CaMgO_full_tmp_2144.gui CaMgO_full_2144.gui
mv CaMgO_full_tmp_2145.f9 CaMgO_full_2145.f9
mv CaMgO_full_tmp_2145.gui CaMgO_full_2145.gui
mv CaMgO_full_tmp_2146.f9 CaMgO_full_2146.f9
mv CaMgO_full_tmp_2146.gui CaMgO_full_2146.gui
mv CaMgO_full_tmp_2147.f9 CaMgO_full_2147.f9
mv CaMgO_full_tmp_2147.gui CaMgO_full_2147.gui
mv CaMgO_full_tmp_2148.f9 CaMgO_full_2148.f9
mv CaMgO_full_tmp_2148.gui CaMgO_full_2148.gui
mv CaMgO_full_tmp_2149.f9 CaMgO_full_2149.f9
mv CaMgO_full_tmp_2149.gui CaMgO_full_2149.gui
mv CaMgO_full_tmp_2150.f9 CaMgO_full_2150.f9
mv CaMgO_full_tmp_2150.gui CaMgO_full_2150.gui
mv CaMgO_full_tmp_2151.f9 CaMgO_full_2151.f9
mv CaMgO_full_tmp_2151.gui CaMgO_full_2151.gui
mv CaMgO_full_tmp_2152.f9 CaMgO_full_2152.f9
mv CaMgO_full_tmp_2152.gui CaMgO_full_2152.gui
mv CaMgO_full_tmp_2153.f9 CaMgO_full_2153.f9
mv CaMgO_full_tmp_2153.gui CaMgO_full_2153.gui
mv CaMgO_full_tmp_2154.f9 CaMgO_full_2154.f9
mv CaMgO_full_tmp_2154.gui CaMgO_full_2154.gui
mv CaMgO_full_tmp_2155.f9 CaMgO_full_2155.f9
mv CaMgO_full_tmp_2155.gui CaMgO_full_2155.gui
mv CaMgO_full_tmp_2156.f9 CaMgO_full_2156.f9
mv CaMgO_full_tmp_2156.gui CaMgO_full_2156.gui
mv CaMgO_full_tmp_2157.f9 CaMgO_full_2157.f9
mv CaMgO_full_tmp_2157.gui CaMgO_full_2157.gui
mv CaMgO_full_tmp_2158.f9 CaMgO_full_2158.f9
mv CaMgO_full_tmp_2158.gui CaMgO_full_2158.gui
mv CaMgO_full_tmp_2159.f9 CaMgO_full_2159.f9
mv CaMgO_full_tmp_2159.gui CaMgO_full_2159.gui
mv CaMgO_full_tmp_2160.f9 CaMgO_full_2160.f9
mv CaMgO_full_tmp_2160.gui CaMgO_full_2160.gui
mv CaMgO_full_tmp_2161.f9 CaMgO_full_2161.f9
mv CaMgO_full_tmp_2161.gui CaMgO_full_2161.gui
mv CaMgO_full_tmp_2162.f9 CaMgO_full_2162.f9
mv CaMgO_full_tmp_2162.gui CaMgO_full_2162.gui
mv CaMgO_full_tmp_2163.f9 CaMgO_full_2163.f9
mv CaMgO_full_tmp_2163.gui CaMgO_full_2163.gui
mv CaMgO_full_tmp_2164.f9 CaMgO_full_2164.f9
mv CaMgO_full_tmp_2164.gui CaMgO_full_2164.gui
mv CaMgO_full_tmp_2165.f9 CaMgO_full_2165.f9
mv CaMgO_full_tmp_2165.gui CaMgO_full_2165.gui
mv CaMgO_full_tmp_2166.f9 CaMgO_full_2166.f9
mv CaMgO_full_tmp_2166.gui CaMgO_full_2166.gui
mv CaMgO_full_tmp_2167.f9 CaMgO_full_2167.f9
mv CaMgO_full_tmp_2167.gui CaMgO_full_2167.gui
mv CaMgO_full_tmp_2168.f9 CaMgO_full_2168.f9
mv CaMgO_full_tmp_2168.gui CaMgO_full_2168.gui
mv CaMgO_full_tmp_2169.f9 CaMgO_full_2169.f9
mv CaMgO_full_tmp_2169.gui CaMgO_full_2169.gui
mv CaMgO_full_tmp_2170.f9 CaMgO_full_2170.f9
mv CaMgO_full_tmp_2170.gui CaMgO_full_2170.gui
mv CaMgO_full_tmp_2171.f9 CaMgO_full_2171.f9
mv CaMgO_full_tmp_2171.gui CaMgO_full_2171.gui
mv CaMgO_full_tmp_2172.f9 CaMgO_full_2172.f9
mv CaMgO_full_tmp_2172.gui CaMgO_full_2172.gui
mv CaMgO_full_tmp_2173.f9 CaMgO_full_2173.f9
mv CaMgO_full_tmp_2173.gui CaMgO_full_2173.gui
mv CaMgO_full_tmp_2174.f9 CaMgO_full_2174.f9
mv CaMgO_full_tmp_2174.gui CaMgO_full_2174.gui
mv CaMgO_full_tmp_2175.f9 CaMgO_full_2175.f9
mv CaMgO_full_tmp_2175.gui CaMgO_full_2175.gui
mv CaMgO_full_tmp_2176.f9 CaMgO_full_2176.f9
mv CaMgO_full_tmp_2176.gui CaMgO_full_2176.gui
mv CaMgO_full_tmp_2177.f9 CaMgO_full_2177.f9
mv CaMgO_full_tmp_2177.gui CaMgO_full_2177.gui
mv CaMgO_full_tmp_2178.f9 CaMgO_full_2178.f9
mv CaMgO_full_tmp_2178.gui CaMgO_full_2178.gui
mv CaMgO_full_tmp_2179.f9 CaMgO_full_2179.f9
mv CaMgO_full_tmp_2179.gui CaMgO_full_2179.gui
mv CaMgO_full_tmp_2180.f9 CaMgO_full_2180.f9
mv CaMgO_full_tmp_2180.gui CaMgO_full_2180.gui
mv CaMgO_full_tmp_2181.f9 CaMgO_full_2181.f9
mv CaMgO_full_tmp_2181.gui CaMgO_full_2181.gui
mv CaMgO_full_tmp_2182.f9 CaMgO_full_2182.f9
mv CaMgO_full_tmp_2182.gui CaMgO_full_2182.gui
mv CaMgO_full_tmp_2183.f9 CaMgO_full_2183.f9
mv CaMgO_full_tmp_2183.gui CaMgO_full_2183.gui
mv CaMgO_full_tmp_2184.f9 CaMgO_full_2184.f9
mv CaMgO_full_tmp_2184.gui CaMgO_full_2184.gui
mv CaMgO_full_tmp_2185.f9 CaMgO_full_2185.f9
mv CaMgO_full_tmp_2185.gui CaMgO_full_2185.gui
mv CaMgO_full_tmp_2186.f9 CaMgO_full_2186.f9
mv CaMgO_full_tmp_2186.gui CaMgO_full_2186.gui
mv CaMgO_full_tmp_2187.f9 CaMgO_full_2187.f9
mv CaMgO_full_tmp_2187.gui CaMgO_full_2187.gui
mv CaMgO_full_tmp_2188.f9 CaMgO_full_2188.f9
mv CaMgO_full_tmp_2188.gui CaMgO_full_2188.gui
mv CaMgO_full_tmp_2189.f9 CaMgO_full_2189.f9
mv CaMgO_full_tmp_2189.gui CaMgO_full_2189.gui
mv CaMgO_full_tmp_2190.f9 CaMgO_full_2190.f9
mv CaMgO_full_tmp_2190.gui CaMgO_full_2190.gui
mv CaMgO_full_tmp_2191.f9 CaMgO_full_2191.f9
mv CaMgO_full_tmp_2191.gui CaMgO_full_2191.gui
mv CaMgO_full_tmp_2192.f9 CaMgO_full_2192.f9
mv CaMgO_full_tmp_2192.gui CaMgO_full_2192.gui
mv CaMgO_full_tmp_2193.f9 CaMgO_full_2193.f9
mv CaMgO_full_tmp_2193.gui CaMgO_full_2193.gui
mv CaMgO_full_tmp_2194.f9 CaMgO_full_2194.f9
mv CaMgO_full_tmp_2194.gui CaMgO_full_2194.gui
mv CaMgO_full_tmp_2195.f9 CaMgO_full_2195.f9
mv CaMgO_full_tmp_2195.gui CaMgO_full_2195.gui
mv CaMgO_full_tmp_2196.f9 CaMgO_full_2196.f9
mv CaMgO_full_tmp_2196.gui CaMgO_full_2196.gui
mv CaMgO_full_tmp_2197.f9 CaMgO_full_2197.f9
mv CaMgO_full_tmp_2197.gui CaMgO_full_2197.gui
mv CaMgO_full_tmp_2198.f9 CaMgO_full_2198.f9
mv CaMgO_full_tmp_2198.gui CaMgO_full_2198.gui
mv CaMgO_full_tmp_2199.f9 CaMgO_full_2199.f9
mv CaMgO_full_tmp_2199.gui CaMgO_full_2199.gui
