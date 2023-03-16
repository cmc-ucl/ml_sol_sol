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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2400


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2499


