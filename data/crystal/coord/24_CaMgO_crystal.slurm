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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2301 CaMgO_coord_2301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2302 CaMgO_coord_2302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2303 CaMgO_coord_2303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2304 CaMgO_coord_2304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2305 CaMgO_coord_2305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2306 CaMgO_coord_2306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2307 CaMgO_coord_2307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2308 CaMgO_coord_2308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2309 CaMgO_coord_2309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2310 CaMgO_coord_2310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2311 CaMgO_coord_2311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2312 CaMgO_coord_2312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2313 CaMgO_coord_2313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2314 CaMgO_coord_2314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2315 CaMgO_coord_2315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2316 CaMgO_coord_2316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2317 CaMgO_coord_2317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2318 CaMgO_coord_2318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2319 CaMgO_coord_2319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2320 CaMgO_coord_2320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2321 CaMgO_coord_2321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2322 CaMgO_coord_2322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2323 CaMgO_coord_2323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2324 CaMgO_coord_2324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2325 CaMgO_coord_2325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2326 CaMgO_coord_2326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2327 CaMgO_coord_2327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2328 CaMgO_coord_2328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2329 CaMgO_coord_2329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2330 CaMgO_coord_2330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2331 CaMgO_coord_2331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2332 CaMgO_coord_2332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2333 CaMgO_coord_2333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2334 CaMgO_coord_2334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2335 CaMgO_coord_2335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2336 CaMgO_coord_2336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2337 CaMgO_coord_2337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2338 CaMgO_coord_2338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2339 CaMgO_coord_2339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2340 CaMgO_coord_2340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2341 CaMgO_coord_2341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2342 CaMgO_coord_2342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2343 CaMgO_coord_2343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2344 CaMgO_coord_2344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2345 CaMgO_coord_2345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2346 CaMgO_coord_2346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2347 CaMgO_coord_2347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2348 CaMgO_coord_2348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2349 CaMgO_coord_2349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2350 CaMgO_coord_2350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2351 CaMgO_coord_2351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2352 CaMgO_coord_2352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2353 CaMgO_coord_2353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2354 CaMgO_coord_2354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2355 CaMgO_coord_2355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2356 CaMgO_coord_2356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2357 CaMgO_coord_2357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2358 CaMgO_coord_2358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2359 CaMgO_coord_2359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2360 CaMgO_coord_2360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2361 CaMgO_coord_2361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2362 CaMgO_coord_2362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2363 CaMgO_coord_2363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2364 CaMgO_coord_2364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2365 CaMgO_coord_2365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2366 CaMgO_coord_2366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2367 CaMgO_coord_2367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2368 CaMgO_coord_2368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2369 CaMgO_coord_2369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2370 CaMgO_coord_2370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2371 CaMgO_coord_2371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2372 CaMgO_coord_2372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2373 CaMgO_coord_2373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2374 CaMgO_coord_2374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2375 CaMgO_coord_2375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2376 CaMgO_coord_2376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2377 CaMgO_coord_2377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2378 CaMgO_coord_2378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2379 CaMgO_coord_2379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2380 CaMgO_coord_2380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2381 CaMgO_coord_2381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2382 CaMgO_coord_2382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2383 CaMgO_coord_2383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2384 CaMgO_coord_2384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2385 CaMgO_coord_2385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2386 CaMgO_coord_2386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2387 CaMgO_coord_2387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2388 CaMgO_coord_2388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2389 CaMgO_coord_2389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2390 CaMgO_coord_2390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2391 CaMgO_coord_2391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2392 CaMgO_coord_2392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2393 CaMgO_coord_2393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2394 CaMgO_coord_2394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2395 CaMgO_coord_2395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2396 CaMgO_coord_2396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2397 CaMgO_coord_2397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2398 CaMgO_coord_2398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2399 CaMgO_coord_2399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2399


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2400 CaMgO_coord_2400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2400


