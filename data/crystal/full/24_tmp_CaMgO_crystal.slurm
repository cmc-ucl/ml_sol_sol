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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2300 CaMgO_full_tmp_2300 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2301 CaMgO_full_tmp_2301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2302 CaMgO_full_tmp_2302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2303 CaMgO_full_tmp_2303 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2304 CaMgO_full_tmp_2304 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2305 CaMgO_full_tmp_2305 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2306 CaMgO_full_tmp_2306 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2307 CaMgO_full_tmp_2307 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2308 CaMgO_full_tmp_2308 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2309 CaMgO_full_tmp_2309 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2310 CaMgO_full_tmp_2310 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2311 CaMgO_full_tmp_2311 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2312 CaMgO_full_tmp_2312 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2313 CaMgO_full_tmp_2313 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2314 CaMgO_full_tmp_2314 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2315 CaMgO_full_tmp_2315 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2316 CaMgO_full_tmp_2316 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2317 CaMgO_full_tmp_2317 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2318 CaMgO_full_tmp_2318 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2319 CaMgO_full_tmp_2319 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2320 CaMgO_full_tmp_2320 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2321 CaMgO_full_tmp_2321 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2322 CaMgO_full_tmp_2322 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2323 CaMgO_full_tmp_2323 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2324 CaMgO_full_tmp_2324 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2325 CaMgO_full_tmp_2325 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2326 CaMgO_full_tmp_2326 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2327 CaMgO_full_tmp_2327 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2328 CaMgO_full_tmp_2328 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2329 CaMgO_full_tmp_2329 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2330 CaMgO_full_tmp_2330 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2331 CaMgO_full_tmp_2331 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2332 CaMgO_full_tmp_2332 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2333 CaMgO_full_tmp_2333 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2334 CaMgO_full_tmp_2334 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2335 CaMgO_full_tmp_2335 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2336 CaMgO_full_tmp_2336 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2337 CaMgO_full_tmp_2337 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2338 CaMgO_full_tmp_2338 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2339 CaMgO_full_tmp_2339 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2340 CaMgO_full_tmp_2340 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2341 CaMgO_full_tmp_2341 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2342 CaMgO_full_tmp_2342 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2343 CaMgO_full_tmp_2343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2344 CaMgO_full_tmp_2344 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2345 CaMgO_full_tmp_2345 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2346 CaMgO_full_tmp_2346 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2347 CaMgO_full_tmp_2347 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2348 CaMgO_full_tmp_2348 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2349 CaMgO_full_tmp_2349 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2350 CaMgO_full_tmp_2350 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2351 CaMgO_full_tmp_2351 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2352 CaMgO_full_tmp_2352 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2353 CaMgO_full_tmp_2353 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2354 CaMgO_full_tmp_2354 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2355 CaMgO_full_tmp_2355 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2356 CaMgO_full_tmp_2356 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2357 CaMgO_full_tmp_2357 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2358 CaMgO_full_tmp_2358 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2359 CaMgO_full_tmp_2359 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2360 CaMgO_full_tmp_2360 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2361 CaMgO_full_tmp_2361 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2362 CaMgO_full_tmp_2362 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2363 CaMgO_full_tmp_2363 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2364 CaMgO_full_tmp_2364 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2365 CaMgO_full_tmp_2365 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2366 CaMgO_full_tmp_2366 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2367 CaMgO_full_tmp_2367 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2368 CaMgO_full_tmp_2368 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2369 CaMgO_full_tmp_2369 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2370 CaMgO_full_tmp_2370 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2371 CaMgO_full_tmp_2371 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2372 CaMgO_full_tmp_2372 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2373 CaMgO_full_tmp_2373 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2374 CaMgO_full_tmp_2374 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2375 CaMgO_full_tmp_2375 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2376 CaMgO_full_tmp_2376 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2377 CaMgO_full_tmp_2377 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2378 CaMgO_full_tmp_2378 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2379 CaMgO_full_tmp_2379 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2380 CaMgO_full_tmp_2380 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2381 CaMgO_full_tmp_2381 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2382 CaMgO_full_tmp_2382 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2383 CaMgO_full_tmp_2383 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2384 CaMgO_full_tmp_2384 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2385 CaMgO_full_tmp_2385 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2386 CaMgO_full_tmp_2386 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2387 CaMgO_full_tmp_2387 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2388 CaMgO_full_tmp_2388 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2389 CaMgO_full_tmp_2389 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2390 CaMgO_full_tmp_2390 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2391 CaMgO_full_tmp_2391 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2392 CaMgO_full_tmp_2392 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2393 CaMgO_full_tmp_2393 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2394 CaMgO_full_tmp_2394 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2395 CaMgO_full_tmp_2395 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2396 CaMgO_full_tmp_2396 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2397 CaMgO_full_tmp_2397 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2398 CaMgO_full_tmp_2398 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2399 CaMgO_full_tmp_2399 &

wait
mv CaMgO_full_tmp_2300.f9 CaMgO_full_2300.f9
mv CaMgO_full_tmp_2300.gui CaMgO_full_2300.gui
mv CaMgO_full_tmp_2301.f9 CaMgO_full_2301.f9
mv CaMgO_full_tmp_2301.gui CaMgO_full_2301.gui
mv CaMgO_full_tmp_2302.f9 CaMgO_full_2302.f9
mv CaMgO_full_tmp_2302.gui CaMgO_full_2302.gui
mv CaMgO_full_tmp_2303.f9 CaMgO_full_2303.f9
mv CaMgO_full_tmp_2303.gui CaMgO_full_2303.gui
mv CaMgO_full_tmp_2304.f9 CaMgO_full_2304.f9
mv CaMgO_full_tmp_2304.gui CaMgO_full_2304.gui
mv CaMgO_full_tmp_2305.f9 CaMgO_full_2305.f9
mv CaMgO_full_tmp_2305.gui CaMgO_full_2305.gui
mv CaMgO_full_tmp_2306.f9 CaMgO_full_2306.f9
mv CaMgO_full_tmp_2306.gui CaMgO_full_2306.gui
mv CaMgO_full_tmp_2307.f9 CaMgO_full_2307.f9
mv CaMgO_full_tmp_2307.gui CaMgO_full_2307.gui
mv CaMgO_full_tmp_2308.f9 CaMgO_full_2308.f9
mv CaMgO_full_tmp_2308.gui CaMgO_full_2308.gui
mv CaMgO_full_tmp_2309.f9 CaMgO_full_2309.f9
mv CaMgO_full_tmp_2309.gui CaMgO_full_2309.gui
mv CaMgO_full_tmp_2310.f9 CaMgO_full_2310.f9
mv CaMgO_full_tmp_2310.gui CaMgO_full_2310.gui
mv CaMgO_full_tmp_2311.f9 CaMgO_full_2311.f9
mv CaMgO_full_tmp_2311.gui CaMgO_full_2311.gui
mv CaMgO_full_tmp_2312.f9 CaMgO_full_2312.f9
mv CaMgO_full_tmp_2312.gui CaMgO_full_2312.gui
mv CaMgO_full_tmp_2313.f9 CaMgO_full_2313.f9
mv CaMgO_full_tmp_2313.gui CaMgO_full_2313.gui
mv CaMgO_full_tmp_2314.f9 CaMgO_full_2314.f9
mv CaMgO_full_tmp_2314.gui CaMgO_full_2314.gui
mv CaMgO_full_tmp_2315.f9 CaMgO_full_2315.f9
mv CaMgO_full_tmp_2315.gui CaMgO_full_2315.gui
mv CaMgO_full_tmp_2316.f9 CaMgO_full_2316.f9
mv CaMgO_full_tmp_2316.gui CaMgO_full_2316.gui
mv CaMgO_full_tmp_2317.f9 CaMgO_full_2317.f9
mv CaMgO_full_tmp_2317.gui CaMgO_full_2317.gui
mv CaMgO_full_tmp_2318.f9 CaMgO_full_2318.f9
mv CaMgO_full_tmp_2318.gui CaMgO_full_2318.gui
mv CaMgO_full_tmp_2319.f9 CaMgO_full_2319.f9
mv CaMgO_full_tmp_2319.gui CaMgO_full_2319.gui
mv CaMgO_full_tmp_2320.f9 CaMgO_full_2320.f9
mv CaMgO_full_tmp_2320.gui CaMgO_full_2320.gui
mv CaMgO_full_tmp_2321.f9 CaMgO_full_2321.f9
mv CaMgO_full_tmp_2321.gui CaMgO_full_2321.gui
mv CaMgO_full_tmp_2322.f9 CaMgO_full_2322.f9
mv CaMgO_full_tmp_2322.gui CaMgO_full_2322.gui
mv CaMgO_full_tmp_2323.f9 CaMgO_full_2323.f9
mv CaMgO_full_tmp_2323.gui CaMgO_full_2323.gui
mv CaMgO_full_tmp_2324.f9 CaMgO_full_2324.f9
mv CaMgO_full_tmp_2324.gui CaMgO_full_2324.gui
mv CaMgO_full_tmp_2325.f9 CaMgO_full_2325.f9
mv CaMgO_full_tmp_2325.gui CaMgO_full_2325.gui
mv CaMgO_full_tmp_2326.f9 CaMgO_full_2326.f9
mv CaMgO_full_tmp_2326.gui CaMgO_full_2326.gui
mv CaMgO_full_tmp_2327.f9 CaMgO_full_2327.f9
mv CaMgO_full_tmp_2327.gui CaMgO_full_2327.gui
mv CaMgO_full_tmp_2328.f9 CaMgO_full_2328.f9
mv CaMgO_full_tmp_2328.gui CaMgO_full_2328.gui
mv CaMgO_full_tmp_2329.f9 CaMgO_full_2329.f9
mv CaMgO_full_tmp_2329.gui CaMgO_full_2329.gui
mv CaMgO_full_tmp_2330.f9 CaMgO_full_2330.f9
mv CaMgO_full_tmp_2330.gui CaMgO_full_2330.gui
mv CaMgO_full_tmp_2331.f9 CaMgO_full_2331.f9
mv CaMgO_full_tmp_2331.gui CaMgO_full_2331.gui
mv CaMgO_full_tmp_2332.f9 CaMgO_full_2332.f9
mv CaMgO_full_tmp_2332.gui CaMgO_full_2332.gui
mv CaMgO_full_tmp_2333.f9 CaMgO_full_2333.f9
mv CaMgO_full_tmp_2333.gui CaMgO_full_2333.gui
mv CaMgO_full_tmp_2334.f9 CaMgO_full_2334.f9
mv CaMgO_full_tmp_2334.gui CaMgO_full_2334.gui
mv CaMgO_full_tmp_2335.f9 CaMgO_full_2335.f9
mv CaMgO_full_tmp_2335.gui CaMgO_full_2335.gui
mv CaMgO_full_tmp_2336.f9 CaMgO_full_2336.f9
mv CaMgO_full_tmp_2336.gui CaMgO_full_2336.gui
mv CaMgO_full_tmp_2337.f9 CaMgO_full_2337.f9
mv CaMgO_full_tmp_2337.gui CaMgO_full_2337.gui
mv CaMgO_full_tmp_2338.f9 CaMgO_full_2338.f9
mv CaMgO_full_tmp_2338.gui CaMgO_full_2338.gui
mv CaMgO_full_tmp_2339.f9 CaMgO_full_2339.f9
mv CaMgO_full_tmp_2339.gui CaMgO_full_2339.gui
mv CaMgO_full_tmp_2340.f9 CaMgO_full_2340.f9
mv CaMgO_full_tmp_2340.gui CaMgO_full_2340.gui
mv CaMgO_full_tmp_2341.f9 CaMgO_full_2341.f9
mv CaMgO_full_tmp_2341.gui CaMgO_full_2341.gui
mv CaMgO_full_tmp_2342.f9 CaMgO_full_2342.f9
mv CaMgO_full_tmp_2342.gui CaMgO_full_2342.gui
mv CaMgO_full_tmp_2343.f9 CaMgO_full_2343.f9
mv CaMgO_full_tmp_2343.gui CaMgO_full_2343.gui
mv CaMgO_full_tmp_2344.f9 CaMgO_full_2344.f9
mv CaMgO_full_tmp_2344.gui CaMgO_full_2344.gui
mv CaMgO_full_tmp_2345.f9 CaMgO_full_2345.f9
mv CaMgO_full_tmp_2345.gui CaMgO_full_2345.gui
mv CaMgO_full_tmp_2346.f9 CaMgO_full_2346.f9
mv CaMgO_full_tmp_2346.gui CaMgO_full_2346.gui
mv CaMgO_full_tmp_2347.f9 CaMgO_full_2347.f9
mv CaMgO_full_tmp_2347.gui CaMgO_full_2347.gui
mv CaMgO_full_tmp_2348.f9 CaMgO_full_2348.f9
mv CaMgO_full_tmp_2348.gui CaMgO_full_2348.gui
mv CaMgO_full_tmp_2349.f9 CaMgO_full_2349.f9
mv CaMgO_full_tmp_2349.gui CaMgO_full_2349.gui
mv CaMgO_full_tmp_2350.f9 CaMgO_full_2350.f9
mv CaMgO_full_tmp_2350.gui CaMgO_full_2350.gui
mv CaMgO_full_tmp_2351.f9 CaMgO_full_2351.f9
mv CaMgO_full_tmp_2351.gui CaMgO_full_2351.gui
mv CaMgO_full_tmp_2352.f9 CaMgO_full_2352.f9
mv CaMgO_full_tmp_2352.gui CaMgO_full_2352.gui
mv CaMgO_full_tmp_2353.f9 CaMgO_full_2353.f9
mv CaMgO_full_tmp_2353.gui CaMgO_full_2353.gui
mv CaMgO_full_tmp_2354.f9 CaMgO_full_2354.f9
mv CaMgO_full_tmp_2354.gui CaMgO_full_2354.gui
mv CaMgO_full_tmp_2355.f9 CaMgO_full_2355.f9
mv CaMgO_full_tmp_2355.gui CaMgO_full_2355.gui
mv CaMgO_full_tmp_2356.f9 CaMgO_full_2356.f9
mv CaMgO_full_tmp_2356.gui CaMgO_full_2356.gui
mv CaMgO_full_tmp_2357.f9 CaMgO_full_2357.f9
mv CaMgO_full_tmp_2357.gui CaMgO_full_2357.gui
mv CaMgO_full_tmp_2358.f9 CaMgO_full_2358.f9
mv CaMgO_full_tmp_2358.gui CaMgO_full_2358.gui
mv CaMgO_full_tmp_2359.f9 CaMgO_full_2359.f9
mv CaMgO_full_tmp_2359.gui CaMgO_full_2359.gui
mv CaMgO_full_tmp_2360.f9 CaMgO_full_2360.f9
mv CaMgO_full_tmp_2360.gui CaMgO_full_2360.gui
mv CaMgO_full_tmp_2361.f9 CaMgO_full_2361.f9
mv CaMgO_full_tmp_2361.gui CaMgO_full_2361.gui
mv CaMgO_full_tmp_2362.f9 CaMgO_full_2362.f9
mv CaMgO_full_tmp_2362.gui CaMgO_full_2362.gui
mv CaMgO_full_tmp_2363.f9 CaMgO_full_2363.f9
mv CaMgO_full_tmp_2363.gui CaMgO_full_2363.gui
mv CaMgO_full_tmp_2364.f9 CaMgO_full_2364.f9
mv CaMgO_full_tmp_2364.gui CaMgO_full_2364.gui
mv CaMgO_full_tmp_2365.f9 CaMgO_full_2365.f9
mv CaMgO_full_tmp_2365.gui CaMgO_full_2365.gui
mv CaMgO_full_tmp_2366.f9 CaMgO_full_2366.f9
mv CaMgO_full_tmp_2366.gui CaMgO_full_2366.gui
mv CaMgO_full_tmp_2367.f9 CaMgO_full_2367.f9
mv CaMgO_full_tmp_2367.gui CaMgO_full_2367.gui
mv CaMgO_full_tmp_2368.f9 CaMgO_full_2368.f9
mv CaMgO_full_tmp_2368.gui CaMgO_full_2368.gui
mv CaMgO_full_tmp_2369.f9 CaMgO_full_2369.f9
mv CaMgO_full_tmp_2369.gui CaMgO_full_2369.gui
mv CaMgO_full_tmp_2370.f9 CaMgO_full_2370.f9
mv CaMgO_full_tmp_2370.gui CaMgO_full_2370.gui
mv CaMgO_full_tmp_2371.f9 CaMgO_full_2371.f9
mv CaMgO_full_tmp_2371.gui CaMgO_full_2371.gui
mv CaMgO_full_tmp_2372.f9 CaMgO_full_2372.f9
mv CaMgO_full_tmp_2372.gui CaMgO_full_2372.gui
mv CaMgO_full_tmp_2373.f9 CaMgO_full_2373.f9
mv CaMgO_full_tmp_2373.gui CaMgO_full_2373.gui
mv CaMgO_full_tmp_2374.f9 CaMgO_full_2374.f9
mv CaMgO_full_tmp_2374.gui CaMgO_full_2374.gui
mv CaMgO_full_tmp_2375.f9 CaMgO_full_2375.f9
mv CaMgO_full_tmp_2375.gui CaMgO_full_2375.gui
mv CaMgO_full_tmp_2376.f9 CaMgO_full_2376.f9
mv CaMgO_full_tmp_2376.gui CaMgO_full_2376.gui
mv CaMgO_full_tmp_2377.f9 CaMgO_full_2377.f9
mv CaMgO_full_tmp_2377.gui CaMgO_full_2377.gui
mv CaMgO_full_tmp_2378.f9 CaMgO_full_2378.f9
mv CaMgO_full_tmp_2378.gui CaMgO_full_2378.gui
mv CaMgO_full_tmp_2379.f9 CaMgO_full_2379.f9
mv CaMgO_full_tmp_2379.gui CaMgO_full_2379.gui
mv CaMgO_full_tmp_2380.f9 CaMgO_full_2380.f9
mv CaMgO_full_tmp_2380.gui CaMgO_full_2380.gui
mv CaMgO_full_tmp_2381.f9 CaMgO_full_2381.f9
mv CaMgO_full_tmp_2381.gui CaMgO_full_2381.gui
mv CaMgO_full_tmp_2382.f9 CaMgO_full_2382.f9
mv CaMgO_full_tmp_2382.gui CaMgO_full_2382.gui
mv CaMgO_full_tmp_2383.f9 CaMgO_full_2383.f9
mv CaMgO_full_tmp_2383.gui CaMgO_full_2383.gui
mv CaMgO_full_tmp_2384.f9 CaMgO_full_2384.f9
mv CaMgO_full_tmp_2384.gui CaMgO_full_2384.gui
mv CaMgO_full_tmp_2385.f9 CaMgO_full_2385.f9
mv CaMgO_full_tmp_2385.gui CaMgO_full_2385.gui
mv CaMgO_full_tmp_2386.f9 CaMgO_full_2386.f9
mv CaMgO_full_tmp_2386.gui CaMgO_full_2386.gui
mv CaMgO_full_tmp_2387.f9 CaMgO_full_2387.f9
mv CaMgO_full_tmp_2387.gui CaMgO_full_2387.gui
mv CaMgO_full_tmp_2388.f9 CaMgO_full_2388.f9
mv CaMgO_full_tmp_2388.gui CaMgO_full_2388.gui
mv CaMgO_full_tmp_2389.f9 CaMgO_full_2389.f9
mv CaMgO_full_tmp_2389.gui CaMgO_full_2389.gui
mv CaMgO_full_tmp_2390.f9 CaMgO_full_2390.f9
mv CaMgO_full_tmp_2390.gui CaMgO_full_2390.gui
mv CaMgO_full_tmp_2391.f9 CaMgO_full_2391.f9
mv CaMgO_full_tmp_2391.gui CaMgO_full_2391.gui
mv CaMgO_full_tmp_2392.f9 CaMgO_full_2392.f9
mv CaMgO_full_tmp_2392.gui CaMgO_full_2392.gui
mv CaMgO_full_tmp_2393.f9 CaMgO_full_2393.f9
mv CaMgO_full_tmp_2393.gui CaMgO_full_2393.gui
mv CaMgO_full_tmp_2394.f9 CaMgO_full_2394.f9
mv CaMgO_full_tmp_2394.gui CaMgO_full_2394.gui
mv CaMgO_full_tmp_2395.f9 CaMgO_full_2395.f9
mv CaMgO_full_tmp_2395.gui CaMgO_full_2395.gui
mv CaMgO_full_tmp_2396.f9 CaMgO_full_2396.f9
mv CaMgO_full_tmp_2396.gui CaMgO_full_2396.gui
mv CaMgO_full_tmp_2397.f9 CaMgO_full_2397.f9
mv CaMgO_full_tmp_2397.gui CaMgO_full_2397.gui
mv CaMgO_full_tmp_2398.f9 CaMgO_full_2398.f9
mv CaMgO_full_tmp_2398.gui CaMgO_full_2398.gui
mv CaMgO_full_tmp_2399.f9 CaMgO_full_2399.f9
mv CaMgO_full_tmp_2399.gui CaMgO_full_2399.gui
