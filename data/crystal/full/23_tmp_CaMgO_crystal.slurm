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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2200 CaMgO_full_tmp_2200 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2201 CaMgO_full_tmp_2201 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2202 CaMgO_full_tmp_2202 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2203 CaMgO_full_tmp_2203 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2204 CaMgO_full_tmp_2204 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2205 CaMgO_full_tmp_2205 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2206 CaMgO_full_tmp_2206 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2207 CaMgO_full_tmp_2207 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2208 CaMgO_full_tmp_2208 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2209 CaMgO_full_tmp_2209 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2210 CaMgO_full_tmp_2210 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2211 CaMgO_full_tmp_2211 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2212 CaMgO_full_tmp_2212 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2213 CaMgO_full_tmp_2213 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2214 CaMgO_full_tmp_2214 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2215 CaMgO_full_tmp_2215 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2216 CaMgO_full_tmp_2216 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2217 CaMgO_full_tmp_2217 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2218 CaMgO_full_tmp_2218 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2219 CaMgO_full_tmp_2219 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2220 CaMgO_full_tmp_2220 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2221 CaMgO_full_tmp_2221 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2222 CaMgO_full_tmp_2222 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2223 CaMgO_full_tmp_2223 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2224 CaMgO_full_tmp_2224 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2225 CaMgO_full_tmp_2225 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2226 CaMgO_full_tmp_2226 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2227 CaMgO_full_tmp_2227 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2228 CaMgO_full_tmp_2228 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2229 CaMgO_full_tmp_2229 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2230 CaMgO_full_tmp_2230 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2231 CaMgO_full_tmp_2231 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2232 CaMgO_full_tmp_2232 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2233 CaMgO_full_tmp_2233 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2234 CaMgO_full_tmp_2234 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2235 CaMgO_full_tmp_2235 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2236 CaMgO_full_tmp_2236 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2237 CaMgO_full_tmp_2237 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2238 CaMgO_full_tmp_2238 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2239 CaMgO_full_tmp_2239 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2240 CaMgO_full_tmp_2240 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2241 CaMgO_full_tmp_2241 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2242 CaMgO_full_tmp_2242 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2243 CaMgO_full_tmp_2243 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2244 CaMgO_full_tmp_2244 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2245 CaMgO_full_tmp_2245 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2246 CaMgO_full_tmp_2246 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2247 CaMgO_full_tmp_2247 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2248 CaMgO_full_tmp_2248 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2249 CaMgO_full_tmp_2249 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2250 CaMgO_full_tmp_2250 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2251 CaMgO_full_tmp_2251 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2252 CaMgO_full_tmp_2252 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2253 CaMgO_full_tmp_2253 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2254 CaMgO_full_tmp_2254 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2255 CaMgO_full_tmp_2255 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2256 CaMgO_full_tmp_2256 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2257 CaMgO_full_tmp_2257 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2258 CaMgO_full_tmp_2258 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2259 CaMgO_full_tmp_2259 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2260 CaMgO_full_tmp_2260 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2261 CaMgO_full_tmp_2261 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2262 CaMgO_full_tmp_2262 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2263 CaMgO_full_tmp_2263 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2264 CaMgO_full_tmp_2264 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2265 CaMgO_full_tmp_2265 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2266 CaMgO_full_tmp_2266 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2267 CaMgO_full_tmp_2267 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2268 CaMgO_full_tmp_2268 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2269 CaMgO_full_tmp_2269 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2270 CaMgO_full_tmp_2270 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2271 CaMgO_full_tmp_2271 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2272 CaMgO_full_tmp_2272 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2273 CaMgO_full_tmp_2273 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2274 CaMgO_full_tmp_2274 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2275 CaMgO_full_tmp_2275 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2276 CaMgO_full_tmp_2276 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2277 CaMgO_full_tmp_2277 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2278 CaMgO_full_tmp_2278 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2279 CaMgO_full_tmp_2279 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2280 CaMgO_full_tmp_2280 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2281 CaMgO_full_tmp_2281 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2282 CaMgO_full_tmp_2282 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2283 CaMgO_full_tmp_2283 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2284 CaMgO_full_tmp_2284 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2285 CaMgO_full_tmp_2285 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2286 CaMgO_full_tmp_2286 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2287 CaMgO_full_tmp_2287 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2288 CaMgO_full_tmp_2288 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2289 CaMgO_full_tmp_2289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2290 CaMgO_full_tmp_2290 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2291 CaMgO_full_tmp_2291 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2292 CaMgO_full_tmp_2292 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2293 CaMgO_full_tmp_2293 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2294 CaMgO_full_tmp_2294 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2295 CaMgO_full_tmp_2295 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2296 CaMgO_full_tmp_2296 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2297 CaMgO_full_tmp_2297 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2298 CaMgO_full_tmp_2298 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2299 CaMgO_full_tmp_2299 &

wait
mv CaMgO_full_tmp_2200.f9 CaMgO_full_2200.f9
mv CaMgO_full_tmp_2200.gui CaMgO_full_2200.gui
mv CaMgO_full_tmp_2201.f9 CaMgO_full_2201.f9
mv CaMgO_full_tmp_2201.gui CaMgO_full_2201.gui
mv CaMgO_full_tmp_2202.f9 CaMgO_full_2202.f9
mv CaMgO_full_tmp_2202.gui CaMgO_full_2202.gui
mv CaMgO_full_tmp_2203.f9 CaMgO_full_2203.f9
mv CaMgO_full_tmp_2203.gui CaMgO_full_2203.gui
mv CaMgO_full_tmp_2204.f9 CaMgO_full_2204.f9
mv CaMgO_full_tmp_2204.gui CaMgO_full_2204.gui
mv CaMgO_full_tmp_2205.f9 CaMgO_full_2205.f9
mv CaMgO_full_tmp_2205.gui CaMgO_full_2205.gui
mv CaMgO_full_tmp_2206.f9 CaMgO_full_2206.f9
mv CaMgO_full_tmp_2206.gui CaMgO_full_2206.gui
mv CaMgO_full_tmp_2207.f9 CaMgO_full_2207.f9
mv CaMgO_full_tmp_2207.gui CaMgO_full_2207.gui
mv CaMgO_full_tmp_2208.f9 CaMgO_full_2208.f9
mv CaMgO_full_tmp_2208.gui CaMgO_full_2208.gui
mv CaMgO_full_tmp_2209.f9 CaMgO_full_2209.f9
mv CaMgO_full_tmp_2209.gui CaMgO_full_2209.gui
mv CaMgO_full_tmp_2210.f9 CaMgO_full_2210.f9
mv CaMgO_full_tmp_2210.gui CaMgO_full_2210.gui
mv CaMgO_full_tmp_2211.f9 CaMgO_full_2211.f9
mv CaMgO_full_tmp_2211.gui CaMgO_full_2211.gui
mv CaMgO_full_tmp_2212.f9 CaMgO_full_2212.f9
mv CaMgO_full_tmp_2212.gui CaMgO_full_2212.gui
mv CaMgO_full_tmp_2213.f9 CaMgO_full_2213.f9
mv CaMgO_full_tmp_2213.gui CaMgO_full_2213.gui
mv CaMgO_full_tmp_2214.f9 CaMgO_full_2214.f9
mv CaMgO_full_tmp_2214.gui CaMgO_full_2214.gui
mv CaMgO_full_tmp_2215.f9 CaMgO_full_2215.f9
mv CaMgO_full_tmp_2215.gui CaMgO_full_2215.gui
mv CaMgO_full_tmp_2216.f9 CaMgO_full_2216.f9
mv CaMgO_full_tmp_2216.gui CaMgO_full_2216.gui
mv CaMgO_full_tmp_2217.f9 CaMgO_full_2217.f9
mv CaMgO_full_tmp_2217.gui CaMgO_full_2217.gui
mv CaMgO_full_tmp_2218.f9 CaMgO_full_2218.f9
mv CaMgO_full_tmp_2218.gui CaMgO_full_2218.gui
mv CaMgO_full_tmp_2219.f9 CaMgO_full_2219.f9
mv CaMgO_full_tmp_2219.gui CaMgO_full_2219.gui
mv CaMgO_full_tmp_2220.f9 CaMgO_full_2220.f9
mv CaMgO_full_tmp_2220.gui CaMgO_full_2220.gui
mv CaMgO_full_tmp_2221.f9 CaMgO_full_2221.f9
mv CaMgO_full_tmp_2221.gui CaMgO_full_2221.gui
mv CaMgO_full_tmp_2222.f9 CaMgO_full_2222.f9
mv CaMgO_full_tmp_2222.gui CaMgO_full_2222.gui
mv CaMgO_full_tmp_2223.f9 CaMgO_full_2223.f9
mv CaMgO_full_tmp_2223.gui CaMgO_full_2223.gui
mv CaMgO_full_tmp_2224.f9 CaMgO_full_2224.f9
mv CaMgO_full_tmp_2224.gui CaMgO_full_2224.gui
mv CaMgO_full_tmp_2225.f9 CaMgO_full_2225.f9
mv CaMgO_full_tmp_2225.gui CaMgO_full_2225.gui
mv CaMgO_full_tmp_2226.f9 CaMgO_full_2226.f9
mv CaMgO_full_tmp_2226.gui CaMgO_full_2226.gui
mv CaMgO_full_tmp_2227.f9 CaMgO_full_2227.f9
mv CaMgO_full_tmp_2227.gui CaMgO_full_2227.gui
mv CaMgO_full_tmp_2228.f9 CaMgO_full_2228.f9
mv CaMgO_full_tmp_2228.gui CaMgO_full_2228.gui
mv CaMgO_full_tmp_2229.f9 CaMgO_full_2229.f9
mv CaMgO_full_tmp_2229.gui CaMgO_full_2229.gui
mv CaMgO_full_tmp_2230.f9 CaMgO_full_2230.f9
mv CaMgO_full_tmp_2230.gui CaMgO_full_2230.gui
mv CaMgO_full_tmp_2231.f9 CaMgO_full_2231.f9
mv CaMgO_full_tmp_2231.gui CaMgO_full_2231.gui
mv CaMgO_full_tmp_2232.f9 CaMgO_full_2232.f9
mv CaMgO_full_tmp_2232.gui CaMgO_full_2232.gui
mv CaMgO_full_tmp_2233.f9 CaMgO_full_2233.f9
mv CaMgO_full_tmp_2233.gui CaMgO_full_2233.gui
mv CaMgO_full_tmp_2234.f9 CaMgO_full_2234.f9
mv CaMgO_full_tmp_2234.gui CaMgO_full_2234.gui
mv CaMgO_full_tmp_2235.f9 CaMgO_full_2235.f9
mv CaMgO_full_tmp_2235.gui CaMgO_full_2235.gui
mv CaMgO_full_tmp_2236.f9 CaMgO_full_2236.f9
mv CaMgO_full_tmp_2236.gui CaMgO_full_2236.gui
mv CaMgO_full_tmp_2237.f9 CaMgO_full_2237.f9
mv CaMgO_full_tmp_2237.gui CaMgO_full_2237.gui
mv CaMgO_full_tmp_2238.f9 CaMgO_full_2238.f9
mv CaMgO_full_tmp_2238.gui CaMgO_full_2238.gui
mv CaMgO_full_tmp_2239.f9 CaMgO_full_2239.f9
mv CaMgO_full_tmp_2239.gui CaMgO_full_2239.gui
mv CaMgO_full_tmp_2240.f9 CaMgO_full_2240.f9
mv CaMgO_full_tmp_2240.gui CaMgO_full_2240.gui
mv CaMgO_full_tmp_2241.f9 CaMgO_full_2241.f9
mv CaMgO_full_tmp_2241.gui CaMgO_full_2241.gui
mv CaMgO_full_tmp_2242.f9 CaMgO_full_2242.f9
mv CaMgO_full_tmp_2242.gui CaMgO_full_2242.gui
mv CaMgO_full_tmp_2243.f9 CaMgO_full_2243.f9
mv CaMgO_full_tmp_2243.gui CaMgO_full_2243.gui
mv CaMgO_full_tmp_2244.f9 CaMgO_full_2244.f9
mv CaMgO_full_tmp_2244.gui CaMgO_full_2244.gui
mv CaMgO_full_tmp_2245.f9 CaMgO_full_2245.f9
mv CaMgO_full_tmp_2245.gui CaMgO_full_2245.gui
mv CaMgO_full_tmp_2246.f9 CaMgO_full_2246.f9
mv CaMgO_full_tmp_2246.gui CaMgO_full_2246.gui
mv CaMgO_full_tmp_2247.f9 CaMgO_full_2247.f9
mv CaMgO_full_tmp_2247.gui CaMgO_full_2247.gui
mv CaMgO_full_tmp_2248.f9 CaMgO_full_2248.f9
mv CaMgO_full_tmp_2248.gui CaMgO_full_2248.gui
mv CaMgO_full_tmp_2249.f9 CaMgO_full_2249.f9
mv CaMgO_full_tmp_2249.gui CaMgO_full_2249.gui
mv CaMgO_full_tmp_2250.f9 CaMgO_full_2250.f9
mv CaMgO_full_tmp_2250.gui CaMgO_full_2250.gui
mv CaMgO_full_tmp_2251.f9 CaMgO_full_2251.f9
mv CaMgO_full_tmp_2251.gui CaMgO_full_2251.gui
mv CaMgO_full_tmp_2252.f9 CaMgO_full_2252.f9
mv CaMgO_full_tmp_2252.gui CaMgO_full_2252.gui
mv CaMgO_full_tmp_2253.f9 CaMgO_full_2253.f9
mv CaMgO_full_tmp_2253.gui CaMgO_full_2253.gui
mv CaMgO_full_tmp_2254.f9 CaMgO_full_2254.f9
mv CaMgO_full_tmp_2254.gui CaMgO_full_2254.gui
mv CaMgO_full_tmp_2255.f9 CaMgO_full_2255.f9
mv CaMgO_full_tmp_2255.gui CaMgO_full_2255.gui
mv CaMgO_full_tmp_2256.f9 CaMgO_full_2256.f9
mv CaMgO_full_tmp_2256.gui CaMgO_full_2256.gui
mv CaMgO_full_tmp_2257.f9 CaMgO_full_2257.f9
mv CaMgO_full_tmp_2257.gui CaMgO_full_2257.gui
mv CaMgO_full_tmp_2258.f9 CaMgO_full_2258.f9
mv CaMgO_full_tmp_2258.gui CaMgO_full_2258.gui
mv CaMgO_full_tmp_2259.f9 CaMgO_full_2259.f9
mv CaMgO_full_tmp_2259.gui CaMgO_full_2259.gui
mv CaMgO_full_tmp_2260.f9 CaMgO_full_2260.f9
mv CaMgO_full_tmp_2260.gui CaMgO_full_2260.gui
mv CaMgO_full_tmp_2261.f9 CaMgO_full_2261.f9
mv CaMgO_full_tmp_2261.gui CaMgO_full_2261.gui
mv CaMgO_full_tmp_2262.f9 CaMgO_full_2262.f9
mv CaMgO_full_tmp_2262.gui CaMgO_full_2262.gui
mv CaMgO_full_tmp_2263.f9 CaMgO_full_2263.f9
mv CaMgO_full_tmp_2263.gui CaMgO_full_2263.gui
mv CaMgO_full_tmp_2264.f9 CaMgO_full_2264.f9
mv CaMgO_full_tmp_2264.gui CaMgO_full_2264.gui
mv CaMgO_full_tmp_2265.f9 CaMgO_full_2265.f9
mv CaMgO_full_tmp_2265.gui CaMgO_full_2265.gui
mv CaMgO_full_tmp_2266.f9 CaMgO_full_2266.f9
mv CaMgO_full_tmp_2266.gui CaMgO_full_2266.gui
mv CaMgO_full_tmp_2267.f9 CaMgO_full_2267.f9
mv CaMgO_full_tmp_2267.gui CaMgO_full_2267.gui
mv CaMgO_full_tmp_2268.f9 CaMgO_full_2268.f9
mv CaMgO_full_tmp_2268.gui CaMgO_full_2268.gui
mv CaMgO_full_tmp_2269.f9 CaMgO_full_2269.f9
mv CaMgO_full_tmp_2269.gui CaMgO_full_2269.gui
mv CaMgO_full_tmp_2270.f9 CaMgO_full_2270.f9
mv CaMgO_full_tmp_2270.gui CaMgO_full_2270.gui
mv CaMgO_full_tmp_2271.f9 CaMgO_full_2271.f9
mv CaMgO_full_tmp_2271.gui CaMgO_full_2271.gui
mv CaMgO_full_tmp_2272.f9 CaMgO_full_2272.f9
mv CaMgO_full_tmp_2272.gui CaMgO_full_2272.gui
mv CaMgO_full_tmp_2273.f9 CaMgO_full_2273.f9
mv CaMgO_full_tmp_2273.gui CaMgO_full_2273.gui
mv CaMgO_full_tmp_2274.f9 CaMgO_full_2274.f9
mv CaMgO_full_tmp_2274.gui CaMgO_full_2274.gui
mv CaMgO_full_tmp_2275.f9 CaMgO_full_2275.f9
mv CaMgO_full_tmp_2275.gui CaMgO_full_2275.gui
mv CaMgO_full_tmp_2276.f9 CaMgO_full_2276.f9
mv CaMgO_full_tmp_2276.gui CaMgO_full_2276.gui
mv CaMgO_full_tmp_2277.f9 CaMgO_full_2277.f9
mv CaMgO_full_tmp_2277.gui CaMgO_full_2277.gui
mv CaMgO_full_tmp_2278.f9 CaMgO_full_2278.f9
mv CaMgO_full_tmp_2278.gui CaMgO_full_2278.gui
mv CaMgO_full_tmp_2279.f9 CaMgO_full_2279.f9
mv CaMgO_full_tmp_2279.gui CaMgO_full_2279.gui
mv CaMgO_full_tmp_2280.f9 CaMgO_full_2280.f9
mv CaMgO_full_tmp_2280.gui CaMgO_full_2280.gui
mv CaMgO_full_tmp_2281.f9 CaMgO_full_2281.f9
mv CaMgO_full_tmp_2281.gui CaMgO_full_2281.gui
mv CaMgO_full_tmp_2282.f9 CaMgO_full_2282.f9
mv CaMgO_full_tmp_2282.gui CaMgO_full_2282.gui
mv CaMgO_full_tmp_2283.f9 CaMgO_full_2283.f9
mv CaMgO_full_tmp_2283.gui CaMgO_full_2283.gui
mv CaMgO_full_tmp_2284.f9 CaMgO_full_2284.f9
mv CaMgO_full_tmp_2284.gui CaMgO_full_2284.gui
mv CaMgO_full_tmp_2285.f9 CaMgO_full_2285.f9
mv CaMgO_full_tmp_2285.gui CaMgO_full_2285.gui
mv CaMgO_full_tmp_2286.f9 CaMgO_full_2286.f9
mv CaMgO_full_tmp_2286.gui CaMgO_full_2286.gui
mv CaMgO_full_tmp_2287.f9 CaMgO_full_2287.f9
mv CaMgO_full_tmp_2287.gui CaMgO_full_2287.gui
mv CaMgO_full_tmp_2288.f9 CaMgO_full_2288.f9
mv CaMgO_full_tmp_2288.gui CaMgO_full_2288.gui
mv CaMgO_full_tmp_2289.f9 CaMgO_full_2289.f9
mv CaMgO_full_tmp_2289.gui CaMgO_full_2289.gui
mv CaMgO_full_tmp_2290.f9 CaMgO_full_2290.f9
mv CaMgO_full_tmp_2290.gui CaMgO_full_2290.gui
mv CaMgO_full_tmp_2291.f9 CaMgO_full_2291.f9
mv CaMgO_full_tmp_2291.gui CaMgO_full_2291.gui
mv CaMgO_full_tmp_2292.f9 CaMgO_full_2292.f9
mv CaMgO_full_tmp_2292.gui CaMgO_full_2292.gui
mv CaMgO_full_tmp_2293.f9 CaMgO_full_2293.f9
mv CaMgO_full_tmp_2293.gui CaMgO_full_2293.gui
mv CaMgO_full_tmp_2294.f9 CaMgO_full_2294.f9
mv CaMgO_full_tmp_2294.gui CaMgO_full_2294.gui
mv CaMgO_full_tmp_2295.f9 CaMgO_full_2295.f9
mv CaMgO_full_tmp_2295.gui CaMgO_full_2295.gui
mv CaMgO_full_tmp_2296.f9 CaMgO_full_2296.f9
mv CaMgO_full_tmp_2296.gui CaMgO_full_2296.gui
mv CaMgO_full_tmp_2297.f9 CaMgO_full_2297.f9
mv CaMgO_full_tmp_2297.gui CaMgO_full_2297.gui
mv CaMgO_full_tmp_2298.f9 CaMgO_full_2298.f9
mv CaMgO_full_tmp_2298.gui CaMgO_full_2298.gui
mv CaMgO_full_tmp_2299.f9 CaMgO_full_2299.f9
mv CaMgO_full_tmp_2299.gui CaMgO_full_2299.gui
