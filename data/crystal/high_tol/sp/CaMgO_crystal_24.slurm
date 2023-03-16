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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2300


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2399


