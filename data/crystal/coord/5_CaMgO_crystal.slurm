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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_401 CaMgO_coord_401
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_401


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_402 CaMgO_coord_402
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_402


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_403 CaMgO_coord_403
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_403


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_404 CaMgO_coord_404
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_404


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_405 CaMgO_coord_405
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_405


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_406 CaMgO_coord_406
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_406


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_407 CaMgO_coord_407
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_407


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_408 CaMgO_coord_408
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_408


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_409 CaMgO_coord_409
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_409


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_410 CaMgO_coord_410
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_410


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_411 CaMgO_coord_411
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_411


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_412 CaMgO_coord_412
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_412


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_413 CaMgO_coord_413
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_413


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_414 CaMgO_coord_414
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_414


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_415 CaMgO_coord_415
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_415


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_416 CaMgO_coord_416
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_416


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_417 CaMgO_coord_417
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_417


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_418 CaMgO_coord_418
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_418


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_419 CaMgO_coord_419
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_419


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_420 CaMgO_coord_420
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_420


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_421 CaMgO_coord_421
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_421


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_422 CaMgO_coord_422
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_422


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_423 CaMgO_coord_423
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_423


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_424 CaMgO_coord_424
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_424


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_425 CaMgO_coord_425
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_425


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_426 CaMgO_coord_426
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_426


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_427 CaMgO_coord_427
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_427


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_428 CaMgO_coord_428
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_428


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_429 CaMgO_coord_429
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_429


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_430 CaMgO_coord_430
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_430


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_431 CaMgO_coord_431
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_431


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_432 CaMgO_coord_432
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_432


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_433 CaMgO_coord_433
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_433


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_434 CaMgO_coord_434
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_434


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_435 CaMgO_coord_435
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_435


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_436 CaMgO_coord_436
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_436


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_437 CaMgO_coord_437
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_437


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_438 CaMgO_coord_438
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_438


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_439 CaMgO_coord_439
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_439


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_440 CaMgO_coord_440
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_440


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_441 CaMgO_coord_441
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_441


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_442 CaMgO_coord_442
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_442


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_443 CaMgO_coord_443
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_443


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_444 CaMgO_coord_444
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_444


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_445 CaMgO_coord_445
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_445


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_446 CaMgO_coord_446
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_446


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_447 CaMgO_coord_447
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_447


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_448 CaMgO_coord_448
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_448


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_449 CaMgO_coord_449
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_449


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_450 CaMgO_coord_450
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_450


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_451 CaMgO_coord_451
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_451


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_452 CaMgO_coord_452
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_452


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_453 CaMgO_coord_453
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_453


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_454 CaMgO_coord_454
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_454


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_455 CaMgO_coord_455
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_455


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_456 CaMgO_coord_456
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_456


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_457 CaMgO_coord_457
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_457


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_458 CaMgO_coord_458
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_458


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_459 CaMgO_coord_459
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_459


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_460 CaMgO_coord_460
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_460


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_461 CaMgO_coord_461
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_461


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_462 CaMgO_coord_462
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_462


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_463 CaMgO_coord_463
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_463


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_464 CaMgO_coord_464
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_464


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_465 CaMgO_coord_465
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_465


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_466 CaMgO_coord_466
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_466


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_467 CaMgO_coord_467
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_467


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_468 CaMgO_coord_468
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_468


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_469 CaMgO_coord_469
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_469


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_470 CaMgO_coord_470
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_470


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_471 CaMgO_coord_471
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_471


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_472 CaMgO_coord_472
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_472


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_473 CaMgO_coord_473
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_473


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_474 CaMgO_coord_474
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_474


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_475 CaMgO_coord_475
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_475


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_476 CaMgO_coord_476
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_476


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_477 CaMgO_coord_477
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_477


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_478 CaMgO_coord_478
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_478


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_479 CaMgO_coord_479
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_479


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_480 CaMgO_coord_480
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_480


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_481 CaMgO_coord_481
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_481


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_482 CaMgO_coord_482
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_482


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_483 CaMgO_coord_483
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_483


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_484 CaMgO_coord_484
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_484


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_485 CaMgO_coord_485
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_485


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_486 CaMgO_coord_486
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_486


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_487 CaMgO_coord_487
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_487


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_488 CaMgO_coord_488
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_488


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_489 CaMgO_coord_489
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_489


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_490 CaMgO_coord_490
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_490


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_491 CaMgO_coord_491
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_491


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_492 CaMgO_coord_492
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_492


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_493 CaMgO_coord_493
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_493


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_494 CaMgO_coord_494
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_494


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_495 CaMgO_coord_495
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_495


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_496 CaMgO_coord_496
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_496


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_497 CaMgO_coord_497
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_497


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_498 CaMgO_coord_498
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_498


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_499 CaMgO_coord_499
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_499


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_500 CaMgO_coord_500
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_500


