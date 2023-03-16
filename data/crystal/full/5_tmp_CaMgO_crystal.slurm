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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_400 CaMgO_full_tmp_400 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_401 CaMgO_full_tmp_401 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_402 CaMgO_full_tmp_402 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_403 CaMgO_full_tmp_403 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_404 CaMgO_full_tmp_404 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_405 CaMgO_full_tmp_405 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_406 CaMgO_full_tmp_406 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_407 CaMgO_full_tmp_407 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_408 CaMgO_full_tmp_408 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_409 CaMgO_full_tmp_409 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_410 CaMgO_full_tmp_410 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_411 CaMgO_full_tmp_411 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_412 CaMgO_full_tmp_412 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_413 CaMgO_full_tmp_413 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_414 CaMgO_full_tmp_414 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_415 CaMgO_full_tmp_415 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_416 CaMgO_full_tmp_416 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_417 CaMgO_full_tmp_417 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_418 CaMgO_full_tmp_418 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_419 CaMgO_full_tmp_419 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_420 CaMgO_full_tmp_420 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_421 CaMgO_full_tmp_421 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_422 CaMgO_full_tmp_422 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_423 CaMgO_full_tmp_423 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_424 CaMgO_full_tmp_424 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_425 CaMgO_full_tmp_425 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_426 CaMgO_full_tmp_426 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_427 CaMgO_full_tmp_427 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_428 CaMgO_full_tmp_428 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_429 CaMgO_full_tmp_429 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_430 CaMgO_full_tmp_430 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_431 CaMgO_full_tmp_431 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_432 CaMgO_full_tmp_432 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_433 CaMgO_full_tmp_433 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_434 CaMgO_full_tmp_434 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_435 CaMgO_full_tmp_435 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_436 CaMgO_full_tmp_436 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_437 CaMgO_full_tmp_437 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_438 CaMgO_full_tmp_438 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_439 CaMgO_full_tmp_439 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_440 CaMgO_full_tmp_440 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_441 CaMgO_full_tmp_441 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_442 CaMgO_full_tmp_442 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_443 CaMgO_full_tmp_443 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_444 CaMgO_full_tmp_444 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_445 CaMgO_full_tmp_445 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_446 CaMgO_full_tmp_446 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_447 CaMgO_full_tmp_447 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_448 CaMgO_full_tmp_448 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_449 CaMgO_full_tmp_449 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_450 CaMgO_full_tmp_450 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_451 CaMgO_full_tmp_451 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_452 CaMgO_full_tmp_452 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_453 CaMgO_full_tmp_453 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_454 CaMgO_full_tmp_454 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_455 CaMgO_full_tmp_455 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_456 CaMgO_full_tmp_456 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_457 CaMgO_full_tmp_457 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_458 CaMgO_full_tmp_458 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_459 CaMgO_full_tmp_459 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_460 CaMgO_full_tmp_460 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_461 CaMgO_full_tmp_461 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_462 CaMgO_full_tmp_462 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_463 CaMgO_full_tmp_463 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_464 CaMgO_full_tmp_464 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_465 CaMgO_full_tmp_465 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_466 CaMgO_full_tmp_466 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_467 CaMgO_full_tmp_467 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_468 CaMgO_full_tmp_468 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_469 CaMgO_full_tmp_469 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_470 CaMgO_full_tmp_470 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_471 CaMgO_full_tmp_471 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_472 CaMgO_full_tmp_472 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_473 CaMgO_full_tmp_473 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_474 CaMgO_full_tmp_474 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_475 CaMgO_full_tmp_475 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_476 CaMgO_full_tmp_476 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_477 CaMgO_full_tmp_477 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_478 CaMgO_full_tmp_478 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_479 CaMgO_full_tmp_479 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_480 CaMgO_full_tmp_480 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_481 CaMgO_full_tmp_481 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_482 CaMgO_full_tmp_482 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_483 CaMgO_full_tmp_483 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_484 CaMgO_full_tmp_484 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_485 CaMgO_full_tmp_485 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_486 CaMgO_full_tmp_486 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_487 CaMgO_full_tmp_487 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_488 CaMgO_full_tmp_488 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_489 CaMgO_full_tmp_489 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_490 CaMgO_full_tmp_490 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_491 CaMgO_full_tmp_491 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_492 CaMgO_full_tmp_492 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_493 CaMgO_full_tmp_493 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_494 CaMgO_full_tmp_494 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_495 CaMgO_full_tmp_495 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_496 CaMgO_full_tmp_496 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_497 CaMgO_full_tmp_497 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_498 CaMgO_full_tmp_498 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_499 CaMgO_full_tmp_499 &

wait
mv CaMgO_full_tmp_400.f9 CaMgO_full_400.f9
mv CaMgO_full_tmp_400.gui CaMgO_full_400.gui
mv CaMgO_full_tmp_401.f9 CaMgO_full_401.f9
mv CaMgO_full_tmp_401.gui CaMgO_full_401.gui
mv CaMgO_full_tmp_402.f9 CaMgO_full_402.f9
mv CaMgO_full_tmp_402.gui CaMgO_full_402.gui
mv CaMgO_full_tmp_403.f9 CaMgO_full_403.f9
mv CaMgO_full_tmp_403.gui CaMgO_full_403.gui
mv CaMgO_full_tmp_404.f9 CaMgO_full_404.f9
mv CaMgO_full_tmp_404.gui CaMgO_full_404.gui
mv CaMgO_full_tmp_405.f9 CaMgO_full_405.f9
mv CaMgO_full_tmp_405.gui CaMgO_full_405.gui
mv CaMgO_full_tmp_406.f9 CaMgO_full_406.f9
mv CaMgO_full_tmp_406.gui CaMgO_full_406.gui
mv CaMgO_full_tmp_407.f9 CaMgO_full_407.f9
mv CaMgO_full_tmp_407.gui CaMgO_full_407.gui
mv CaMgO_full_tmp_408.f9 CaMgO_full_408.f9
mv CaMgO_full_tmp_408.gui CaMgO_full_408.gui
mv CaMgO_full_tmp_409.f9 CaMgO_full_409.f9
mv CaMgO_full_tmp_409.gui CaMgO_full_409.gui
mv CaMgO_full_tmp_410.f9 CaMgO_full_410.f9
mv CaMgO_full_tmp_410.gui CaMgO_full_410.gui
mv CaMgO_full_tmp_411.f9 CaMgO_full_411.f9
mv CaMgO_full_tmp_411.gui CaMgO_full_411.gui
mv CaMgO_full_tmp_412.f9 CaMgO_full_412.f9
mv CaMgO_full_tmp_412.gui CaMgO_full_412.gui
mv CaMgO_full_tmp_413.f9 CaMgO_full_413.f9
mv CaMgO_full_tmp_413.gui CaMgO_full_413.gui
mv CaMgO_full_tmp_414.f9 CaMgO_full_414.f9
mv CaMgO_full_tmp_414.gui CaMgO_full_414.gui
mv CaMgO_full_tmp_415.f9 CaMgO_full_415.f9
mv CaMgO_full_tmp_415.gui CaMgO_full_415.gui
mv CaMgO_full_tmp_416.f9 CaMgO_full_416.f9
mv CaMgO_full_tmp_416.gui CaMgO_full_416.gui
mv CaMgO_full_tmp_417.f9 CaMgO_full_417.f9
mv CaMgO_full_tmp_417.gui CaMgO_full_417.gui
mv CaMgO_full_tmp_418.f9 CaMgO_full_418.f9
mv CaMgO_full_tmp_418.gui CaMgO_full_418.gui
mv CaMgO_full_tmp_419.f9 CaMgO_full_419.f9
mv CaMgO_full_tmp_419.gui CaMgO_full_419.gui
mv CaMgO_full_tmp_420.f9 CaMgO_full_420.f9
mv CaMgO_full_tmp_420.gui CaMgO_full_420.gui
mv CaMgO_full_tmp_421.f9 CaMgO_full_421.f9
mv CaMgO_full_tmp_421.gui CaMgO_full_421.gui
mv CaMgO_full_tmp_422.f9 CaMgO_full_422.f9
mv CaMgO_full_tmp_422.gui CaMgO_full_422.gui
mv CaMgO_full_tmp_423.f9 CaMgO_full_423.f9
mv CaMgO_full_tmp_423.gui CaMgO_full_423.gui
mv CaMgO_full_tmp_424.f9 CaMgO_full_424.f9
mv CaMgO_full_tmp_424.gui CaMgO_full_424.gui
mv CaMgO_full_tmp_425.f9 CaMgO_full_425.f9
mv CaMgO_full_tmp_425.gui CaMgO_full_425.gui
mv CaMgO_full_tmp_426.f9 CaMgO_full_426.f9
mv CaMgO_full_tmp_426.gui CaMgO_full_426.gui
mv CaMgO_full_tmp_427.f9 CaMgO_full_427.f9
mv CaMgO_full_tmp_427.gui CaMgO_full_427.gui
mv CaMgO_full_tmp_428.f9 CaMgO_full_428.f9
mv CaMgO_full_tmp_428.gui CaMgO_full_428.gui
mv CaMgO_full_tmp_429.f9 CaMgO_full_429.f9
mv CaMgO_full_tmp_429.gui CaMgO_full_429.gui
mv CaMgO_full_tmp_430.f9 CaMgO_full_430.f9
mv CaMgO_full_tmp_430.gui CaMgO_full_430.gui
mv CaMgO_full_tmp_431.f9 CaMgO_full_431.f9
mv CaMgO_full_tmp_431.gui CaMgO_full_431.gui
mv CaMgO_full_tmp_432.f9 CaMgO_full_432.f9
mv CaMgO_full_tmp_432.gui CaMgO_full_432.gui
mv CaMgO_full_tmp_433.f9 CaMgO_full_433.f9
mv CaMgO_full_tmp_433.gui CaMgO_full_433.gui
mv CaMgO_full_tmp_434.f9 CaMgO_full_434.f9
mv CaMgO_full_tmp_434.gui CaMgO_full_434.gui
mv CaMgO_full_tmp_435.f9 CaMgO_full_435.f9
mv CaMgO_full_tmp_435.gui CaMgO_full_435.gui
mv CaMgO_full_tmp_436.f9 CaMgO_full_436.f9
mv CaMgO_full_tmp_436.gui CaMgO_full_436.gui
mv CaMgO_full_tmp_437.f9 CaMgO_full_437.f9
mv CaMgO_full_tmp_437.gui CaMgO_full_437.gui
mv CaMgO_full_tmp_438.f9 CaMgO_full_438.f9
mv CaMgO_full_tmp_438.gui CaMgO_full_438.gui
mv CaMgO_full_tmp_439.f9 CaMgO_full_439.f9
mv CaMgO_full_tmp_439.gui CaMgO_full_439.gui
mv CaMgO_full_tmp_440.f9 CaMgO_full_440.f9
mv CaMgO_full_tmp_440.gui CaMgO_full_440.gui
mv CaMgO_full_tmp_441.f9 CaMgO_full_441.f9
mv CaMgO_full_tmp_441.gui CaMgO_full_441.gui
mv CaMgO_full_tmp_442.f9 CaMgO_full_442.f9
mv CaMgO_full_tmp_442.gui CaMgO_full_442.gui
mv CaMgO_full_tmp_443.f9 CaMgO_full_443.f9
mv CaMgO_full_tmp_443.gui CaMgO_full_443.gui
mv CaMgO_full_tmp_444.f9 CaMgO_full_444.f9
mv CaMgO_full_tmp_444.gui CaMgO_full_444.gui
mv CaMgO_full_tmp_445.f9 CaMgO_full_445.f9
mv CaMgO_full_tmp_445.gui CaMgO_full_445.gui
mv CaMgO_full_tmp_446.f9 CaMgO_full_446.f9
mv CaMgO_full_tmp_446.gui CaMgO_full_446.gui
mv CaMgO_full_tmp_447.f9 CaMgO_full_447.f9
mv CaMgO_full_tmp_447.gui CaMgO_full_447.gui
mv CaMgO_full_tmp_448.f9 CaMgO_full_448.f9
mv CaMgO_full_tmp_448.gui CaMgO_full_448.gui
mv CaMgO_full_tmp_449.f9 CaMgO_full_449.f9
mv CaMgO_full_tmp_449.gui CaMgO_full_449.gui
mv CaMgO_full_tmp_450.f9 CaMgO_full_450.f9
mv CaMgO_full_tmp_450.gui CaMgO_full_450.gui
mv CaMgO_full_tmp_451.f9 CaMgO_full_451.f9
mv CaMgO_full_tmp_451.gui CaMgO_full_451.gui
mv CaMgO_full_tmp_452.f9 CaMgO_full_452.f9
mv CaMgO_full_tmp_452.gui CaMgO_full_452.gui
mv CaMgO_full_tmp_453.f9 CaMgO_full_453.f9
mv CaMgO_full_tmp_453.gui CaMgO_full_453.gui
mv CaMgO_full_tmp_454.f9 CaMgO_full_454.f9
mv CaMgO_full_tmp_454.gui CaMgO_full_454.gui
mv CaMgO_full_tmp_455.f9 CaMgO_full_455.f9
mv CaMgO_full_tmp_455.gui CaMgO_full_455.gui
mv CaMgO_full_tmp_456.f9 CaMgO_full_456.f9
mv CaMgO_full_tmp_456.gui CaMgO_full_456.gui
mv CaMgO_full_tmp_457.f9 CaMgO_full_457.f9
mv CaMgO_full_tmp_457.gui CaMgO_full_457.gui
mv CaMgO_full_tmp_458.f9 CaMgO_full_458.f9
mv CaMgO_full_tmp_458.gui CaMgO_full_458.gui
mv CaMgO_full_tmp_459.f9 CaMgO_full_459.f9
mv CaMgO_full_tmp_459.gui CaMgO_full_459.gui
mv CaMgO_full_tmp_460.f9 CaMgO_full_460.f9
mv CaMgO_full_tmp_460.gui CaMgO_full_460.gui
mv CaMgO_full_tmp_461.f9 CaMgO_full_461.f9
mv CaMgO_full_tmp_461.gui CaMgO_full_461.gui
mv CaMgO_full_tmp_462.f9 CaMgO_full_462.f9
mv CaMgO_full_tmp_462.gui CaMgO_full_462.gui
mv CaMgO_full_tmp_463.f9 CaMgO_full_463.f9
mv CaMgO_full_tmp_463.gui CaMgO_full_463.gui
mv CaMgO_full_tmp_464.f9 CaMgO_full_464.f9
mv CaMgO_full_tmp_464.gui CaMgO_full_464.gui
mv CaMgO_full_tmp_465.f9 CaMgO_full_465.f9
mv CaMgO_full_tmp_465.gui CaMgO_full_465.gui
mv CaMgO_full_tmp_466.f9 CaMgO_full_466.f9
mv CaMgO_full_tmp_466.gui CaMgO_full_466.gui
mv CaMgO_full_tmp_467.f9 CaMgO_full_467.f9
mv CaMgO_full_tmp_467.gui CaMgO_full_467.gui
mv CaMgO_full_tmp_468.f9 CaMgO_full_468.f9
mv CaMgO_full_tmp_468.gui CaMgO_full_468.gui
mv CaMgO_full_tmp_469.f9 CaMgO_full_469.f9
mv CaMgO_full_tmp_469.gui CaMgO_full_469.gui
mv CaMgO_full_tmp_470.f9 CaMgO_full_470.f9
mv CaMgO_full_tmp_470.gui CaMgO_full_470.gui
mv CaMgO_full_tmp_471.f9 CaMgO_full_471.f9
mv CaMgO_full_tmp_471.gui CaMgO_full_471.gui
mv CaMgO_full_tmp_472.f9 CaMgO_full_472.f9
mv CaMgO_full_tmp_472.gui CaMgO_full_472.gui
mv CaMgO_full_tmp_473.f9 CaMgO_full_473.f9
mv CaMgO_full_tmp_473.gui CaMgO_full_473.gui
mv CaMgO_full_tmp_474.f9 CaMgO_full_474.f9
mv CaMgO_full_tmp_474.gui CaMgO_full_474.gui
mv CaMgO_full_tmp_475.f9 CaMgO_full_475.f9
mv CaMgO_full_tmp_475.gui CaMgO_full_475.gui
mv CaMgO_full_tmp_476.f9 CaMgO_full_476.f9
mv CaMgO_full_tmp_476.gui CaMgO_full_476.gui
mv CaMgO_full_tmp_477.f9 CaMgO_full_477.f9
mv CaMgO_full_tmp_477.gui CaMgO_full_477.gui
mv CaMgO_full_tmp_478.f9 CaMgO_full_478.f9
mv CaMgO_full_tmp_478.gui CaMgO_full_478.gui
mv CaMgO_full_tmp_479.f9 CaMgO_full_479.f9
mv CaMgO_full_tmp_479.gui CaMgO_full_479.gui
mv CaMgO_full_tmp_480.f9 CaMgO_full_480.f9
mv CaMgO_full_tmp_480.gui CaMgO_full_480.gui
mv CaMgO_full_tmp_481.f9 CaMgO_full_481.f9
mv CaMgO_full_tmp_481.gui CaMgO_full_481.gui
mv CaMgO_full_tmp_482.f9 CaMgO_full_482.f9
mv CaMgO_full_tmp_482.gui CaMgO_full_482.gui
mv CaMgO_full_tmp_483.f9 CaMgO_full_483.f9
mv CaMgO_full_tmp_483.gui CaMgO_full_483.gui
mv CaMgO_full_tmp_484.f9 CaMgO_full_484.f9
mv CaMgO_full_tmp_484.gui CaMgO_full_484.gui
mv CaMgO_full_tmp_485.f9 CaMgO_full_485.f9
mv CaMgO_full_tmp_485.gui CaMgO_full_485.gui
mv CaMgO_full_tmp_486.f9 CaMgO_full_486.f9
mv CaMgO_full_tmp_486.gui CaMgO_full_486.gui
mv CaMgO_full_tmp_487.f9 CaMgO_full_487.f9
mv CaMgO_full_tmp_487.gui CaMgO_full_487.gui
mv CaMgO_full_tmp_488.f9 CaMgO_full_488.f9
mv CaMgO_full_tmp_488.gui CaMgO_full_488.gui
mv CaMgO_full_tmp_489.f9 CaMgO_full_489.f9
mv CaMgO_full_tmp_489.gui CaMgO_full_489.gui
mv CaMgO_full_tmp_490.f9 CaMgO_full_490.f9
mv CaMgO_full_tmp_490.gui CaMgO_full_490.gui
mv CaMgO_full_tmp_491.f9 CaMgO_full_491.f9
mv CaMgO_full_tmp_491.gui CaMgO_full_491.gui
mv CaMgO_full_tmp_492.f9 CaMgO_full_492.f9
mv CaMgO_full_tmp_492.gui CaMgO_full_492.gui
mv CaMgO_full_tmp_493.f9 CaMgO_full_493.f9
mv CaMgO_full_tmp_493.gui CaMgO_full_493.gui
mv CaMgO_full_tmp_494.f9 CaMgO_full_494.f9
mv CaMgO_full_tmp_494.gui CaMgO_full_494.gui
mv CaMgO_full_tmp_495.f9 CaMgO_full_495.f9
mv CaMgO_full_tmp_495.gui CaMgO_full_495.gui
mv CaMgO_full_tmp_496.f9 CaMgO_full_496.f9
mv CaMgO_full_tmp_496.gui CaMgO_full_496.gui
mv CaMgO_full_tmp_497.f9 CaMgO_full_497.f9
mv CaMgO_full_tmp_497.gui CaMgO_full_497.gui
mv CaMgO_full_tmp_498.f9 CaMgO_full_498.f9
mv CaMgO_full_tmp_498.gui CaMgO_full_498.gui
mv CaMgO_full_tmp_499.f9 CaMgO_full_499.f9
mv CaMgO_full_tmp_499.gui CaMgO_full_499.gui
