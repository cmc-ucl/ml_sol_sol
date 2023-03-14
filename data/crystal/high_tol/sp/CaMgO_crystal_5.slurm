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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_400


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_499


