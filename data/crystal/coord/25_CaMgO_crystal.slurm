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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2401 CaMgO_coord_2401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2402 CaMgO_coord_2402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2403 CaMgO_coord_2403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2404 CaMgO_coord_2404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2405 CaMgO_coord_2405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2406 CaMgO_coord_2406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2407 CaMgO_coord_2407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2408 CaMgO_coord_2408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2409 CaMgO_coord_2409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2410 CaMgO_coord_2410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2411 CaMgO_coord_2411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2412 CaMgO_coord_2412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2413 CaMgO_coord_2413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2414 CaMgO_coord_2414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2415 CaMgO_coord_2415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2416 CaMgO_coord_2416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2417 CaMgO_coord_2417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2418 CaMgO_coord_2418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2419 CaMgO_coord_2419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2420 CaMgO_coord_2420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2421 CaMgO_coord_2421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2422 CaMgO_coord_2422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2423 CaMgO_coord_2423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2424 CaMgO_coord_2424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2425 CaMgO_coord_2425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2426 CaMgO_coord_2426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2427 CaMgO_coord_2427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2428 CaMgO_coord_2428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2429 CaMgO_coord_2429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2430 CaMgO_coord_2430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2431 CaMgO_coord_2431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2432 CaMgO_coord_2432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2433 CaMgO_coord_2433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2434 CaMgO_coord_2434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2435 CaMgO_coord_2435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2436 CaMgO_coord_2436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2437 CaMgO_coord_2437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2438 CaMgO_coord_2438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2439 CaMgO_coord_2439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2440 CaMgO_coord_2440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2441 CaMgO_coord_2441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2442 CaMgO_coord_2442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2443 CaMgO_coord_2443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2444 CaMgO_coord_2444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2445 CaMgO_coord_2445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2446 CaMgO_coord_2446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2447 CaMgO_coord_2447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2448 CaMgO_coord_2448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2449 CaMgO_coord_2449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2450 CaMgO_coord_2450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2451 CaMgO_coord_2451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2452 CaMgO_coord_2452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2453 CaMgO_coord_2453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2454 CaMgO_coord_2454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2455 CaMgO_coord_2455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2456 CaMgO_coord_2456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2457 CaMgO_coord_2457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2458 CaMgO_coord_2458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2459 CaMgO_coord_2459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2460 CaMgO_coord_2460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2461 CaMgO_coord_2461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2462 CaMgO_coord_2462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2463 CaMgO_coord_2463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2464 CaMgO_coord_2464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2465 CaMgO_coord_2465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2466 CaMgO_coord_2466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2467 CaMgO_coord_2467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2468 CaMgO_coord_2468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2469 CaMgO_coord_2469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2470 CaMgO_coord_2470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2471 CaMgO_coord_2471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2472 CaMgO_coord_2472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2473 CaMgO_coord_2473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2474 CaMgO_coord_2474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2475 CaMgO_coord_2475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2476 CaMgO_coord_2476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2477 CaMgO_coord_2477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2478 CaMgO_coord_2478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2479 CaMgO_coord_2479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2480 CaMgO_coord_2480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2481 CaMgO_coord_2481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2482 CaMgO_coord_2482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2483 CaMgO_coord_2483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2484 CaMgO_coord_2484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2485 CaMgO_coord_2485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2486 CaMgO_coord_2486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2487 CaMgO_coord_2487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2488 CaMgO_coord_2488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2489 CaMgO_coord_2489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2490 CaMgO_coord_2490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2491 CaMgO_coord_2491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2492 CaMgO_coord_2492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2493 CaMgO_coord_2493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2494 CaMgO_coord_2494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2495 CaMgO_coord_2495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2496 CaMgO_coord_2496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2497 CaMgO_coord_2497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2498 CaMgO_coord_2498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2499 CaMgO_coord_2499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2499


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2500 CaMgO_coord_2500
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2500


