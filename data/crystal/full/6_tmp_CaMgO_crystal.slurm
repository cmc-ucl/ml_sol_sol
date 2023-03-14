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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_500 CaMgO_full_tmp_500 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_501 CaMgO_full_tmp_501 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_502 CaMgO_full_tmp_502 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_503 CaMgO_full_tmp_503 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_504 CaMgO_full_tmp_504 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_505 CaMgO_full_tmp_505 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_506 CaMgO_full_tmp_506 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_507 CaMgO_full_tmp_507 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_508 CaMgO_full_tmp_508 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_509 CaMgO_full_tmp_509 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_510 CaMgO_full_tmp_510 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_511 CaMgO_full_tmp_511 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_512 CaMgO_full_tmp_512 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_513 CaMgO_full_tmp_513 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_514 CaMgO_full_tmp_514 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_515 CaMgO_full_tmp_515 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_516 CaMgO_full_tmp_516 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_517 CaMgO_full_tmp_517 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_518 CaMgO_full_tmp_518 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_519 CaMgO_full_tmp_519 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_520 CaMgO_full_tmp_520 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_521 CaMgO_full_tmp_521 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_522 CaMgO_full_tmp_522 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_523 CaMgO_full_tmp_523 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_524 CaMgO_full_tmp_524 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_525 CaMgO_full_tmp_525 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_526 CaMgO_full_tmp_526 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_527 CaMgO_full_tmp_527 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_528 CaMgO_full_tmp_528 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_529 CaMgO_full_tmp_529 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_530 CaMgO_full_tmp_530 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_531 CaMgO_full_tmp_531 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_532 CaMgO_full_tmp_532 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_533 CaMgO_full_tmp_533 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_534 CaMgO_full_tmp_534 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_535 CaMgO_full_tmp_535 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_536 CaMgO_full_tmp_536 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_537 CaMgO_full_tmp_537 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_538 CaMgO_full_tmp_538 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_539 CaMgO_full_tmp_539 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_540 CaMgO_full_tmp_540 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_541 CaMgO_full_tmp_541 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_542 CaMgO_full_tmp_542 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_543 CaMgO_full_tmp_543 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_544 CaMgO_full_tmp_544 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_545 CaMgO_full_tmp_545 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_546 CaMgO_full_tmp_546 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_547 CaMgO_full_tmp_547 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_548 CaMgO_full_tmp_548 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_549 CaMgO_full_tmp_549 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_550 CaMgO_full_tmp_550 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_551 CaMgO_full_tmp_551 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_552 CaMgO_full_tmp_552 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_553 CaMgO_full_tmp_553 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_554 CaMgO_full_tmp_554 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_555 CaMgO_full_tmp_555 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_556 CaMgO_full_tmp_556 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_557 CaMgO_full_tmp_557 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_558 CaMgO_full_tmp_558 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_559 CaMgO_full_tmp_559 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_560 CaMgO_full_tmp_560 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_561 CaMgO_full_tmp_561 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_562 CaMgO_full_tmp_562 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_563 CaMgO_full_tmp_563 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_564 CaMgO_full_tmp_564 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_565 CaMgO_full_tmp_565 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_566 CaMgO_full_tmp_566 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_567 CaMgO_full_tmp_567 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_568 CaMgO_full_tmp_568 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_569 CaMgO_full_tmp_569 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_570 CaMgO_full_tmp_570 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_571 CaMgO_full_tmp_571 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_572 CaMgO_full_tmp_572 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_573 CaMgO_full_tmp_573 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_574 CaMgO_full_tmp_574 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_575 CaMgO_full_tmp_575 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_576 CaMgO_full_tmp_576 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_577 CaMgO_full_tmp_577 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_578 CaMgO_full_tmp_578 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_579 CaMgO_full_tmp_579 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_580 CaMgO_full_tmp_580 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_581 CaMgO_full_tmp_581 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_582 CaMgO_full_tmp_582 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_583 CaMgO_full_tmp_583 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_584 CaMgO_full_tmp_584 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_585 CaMgO_full_tmp_585 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_586 CaMgO_full_tmp_586 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_587 CaMgO_full_tmp_587 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_588 CaMgO_full_tmp_588 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_589 CaMgO_full_tmp_589 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_590 CaMgO_full_tmp_590 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_591 CaMgO_full_tmp_591 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_592 CaMgO_full_tmp_592 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_593 CaMgO_full_tmp_593 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_594 CaMgO_full_tmp_594 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_595 CaMgO_full_tmp_595 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_596 CaMgO_full_tmp_596 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_597 CaMgO_full_tmp_597 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_598 CaMgO_full_tmp_598 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_599 CaMgO_full_tmp_599 &

wait
mv CaMgO_full_tmp_500.f9 CaMgO_full_500.f9
mv CaMgO_full_tmp_500.gui CaMgO_full_500.gui
mv CaMgO_full_tmp_501.f9 CaMgO_full_501.f9
mv CaMgO_full_tmp_501.gui CaMgO_full_501.gui
mv CaMgO_full_tmp_502.f9 CaMgO_full_502.f9
mv CaMgO_full_tmp_502.gui CaMgO_full_502.gui
mv CaMgO_full_tmp_503.f9 CaMgO_full_503.f9
mv CaMgO_full_tmp_503.gui CaMgO_full_503.gui
mv CaMgO_full_tmp_504.f9 CaMgO_full_504.f9
mv CaMgO_full_tmp_504.gui CaMgO_full_504.gui
mv CaMgO_full_tmp_505.f9 CaMgO_full_505.f9
mv CaMgO_full_tmp_505.gui CaMgO_full_505.gui
mv CaMgO_full_tmp_506.f9 CaMgO_full_506.f9
mv CaMgO_full_tmp_506.gui CaMgO_full_506.gui
mv CaMgO_full_tmp_507.f9 CaMgO_full_507.f9
mv CaMgO_full_tmp_507.gui CaMgO_full_507.gui
mv CaMgO_full_tmp_508.f9 CaMgO_full_508.f9
mv CaMgO_full_tmp_508.gui CaMgO_full_508.gui
mv CaMgO_full_tmp_509.f9 CaMgO_full_509.f9
mv CaMgO_full_tmp_509.gui CaMgO_full_509.gui
mv CaMgO_full_tmp_510.f9 CaMgO_full_510.f9
mv CaMgO_full_tmp_510.gui CaMgO_full_510.gui
mv CaMgO_full_tmp_511.f9 CaMgO_full_511.f9
mv CaMgO_full_tmp_511.gui CaMgO_full_511.gui
mv CaMgO_full_tmp_512.f9 CaMgO_full_512.f9
mv CaMgO_full_tmp_512.gui CaMgO_full_512.gui
mv CaMgO_full_tmp_513.f9 CaMgO_full_513.f9
mv CaMgO_full_tmp_513.gui CaMgO_full_513.gui
mv CaMgO_full_tmp_514.f9 CaMgO_full_514.f9
mv CaMgO_full_tmp_514.gui CaMgO_full_514.gui
mv CaMgO_full_tmp_515.f9 CaMgO_full_515.f9
mv CaMgO_full_tmp_515.gui CaMgO_full_515.gui
mv CaMgO_full_tmp_516.f9 CaMgO_full_516.f9
mv CaMgO_full_tmp_516.gui CaMgO_full_516.gui
mv CaMgO_full_tmp_517.f9 CaMgO_full_517.f9
mv CaMgO_full_tmp_517.gui CaMgO_full_517.gui
mv CaMgO_full_tmp_518.f9 CaMgO_full_518.f9
mv CaMgO_full_tmp_518.gui CaMgO_full_518.gui
mv CaMgO_full_tmp_519.f9 CaMgO_full_519.f9
mv CaMgO_full_tmp_519.gui CaMgO_full_519.gui
mv CaMgO_full_tmp_520.f9 CaMgO_full_520.f9
mv CaMgO_full_tmp_520.gui CaMgO_full_520.gui
mv CaMgO_full_tmp_521.f9 CaMgO_full_521.f9
mv CaMgO_full_tmp_521.gui CaMgO_full_521.gui
mv CaMgO_full_tmp_522.f9 CaMgO_full_522.f9
mv CaMgO_full_tmp_522.gui CaMgO_full_522.gui
mv CaMgO_full_tmp_523.f9 CaMgO_full_523.f9
mv CaMgO_full_tmp_523.gui CaMgO_full_523.gui
mv CaMgO_full_tmp_524.f9 CaMgO_full_524.f9
mv CaMgO_full_tmp_524.gui CaMgO_full_524.gui
mv CaMgO_full_tmp_525.f9 CaMgO_full_525.f9
mv CaMgO_full_tmp_525.gui CaMgO_full_525.gui
mv CaMgO_full_tmp_526.f9 CaMgO_full_526.f9
mv CaMgO_full_tmp_526.gui CaMgO_full_526.gui
mv CaMgO_full_tmp_527.f9 CaMgO_full_527.f9
mv CaMgO_full_tmp_527.gui CaMgO_full_527.gui
mv CaMgO_full_tmp_528.f9 CaMgO_full_528.f9
mv CaMgO_full_tmp_528.gui CaMgO_full_528.gui
mv CaMgO_full_tmp_529.f9 CaMgO_full_529.f9
mv CaMgO_full_tmp_529.gui CaMgO_full_529.gui
mv CaMgO_full_tmp_530.f9 CaMgO_full_530.f9
mv CaMgO_full_tmp_530.gui CaMgO_full_530.gui
mv CaMgO_full_tmp_531.f9 CaMgO_full_531.f9
mv CaMgO_full_tmp_531.gui CaMgO_full_531.gui
mv CaMgO_full_tmp_532.f9 CaMgO_full_532.f9
mv CaMgO_full_tmp_532.gui CaMgO_full_532.gui
mv CaMgO_full_tmp_533.f9 CaMgO_full_533.f9
mv CaMgO_full_tmp_533.gui CaMgO_full_533.gui
mv CaMgO_full_tmp_534.f9 CaMgO_full_534.f9
mv CaMgO_full_tmp_534.gui CaMgO_full_534.gui
mv CaMgO_full_tmp_535.f9 CaMgO_full_535.f9
mv CaMgO_full_tmp_535.gui CaMgO_full_535.gui
mv CaMgO_full_tmp_536.f9 CaMgO_full_536.f9
mv CaMgO_full_tmp_536.gui CaMgO_full_536.gui
mv CaMgO_full_tmp_537.f9 CaMgO_full_537.f9
mv CaMgO_full_tmp_537.gui CaMgO_full_537.gui
mv CaMgO_full_tmp_538.f9 CaMgO_full_538.f9
mv CaMgO_full_tmp_538.gui CaMgO_full_538.gui
mv CaMgO_full_tmp_539.f9 CaMgO_full_539.f9
mv CaMgO_full_tmp_539.gui CaMgO_full_539.gui
mv CaMgO_full_tmp_540.f9 CaMgO_full_540.f9
mv CaMgO_full_tmp_540.gui CaMgO_full_540.gui
mv CaMgO_full_tmp_541.f9 CaMgO_full_541.f9
mv CaMgO_full_tmp_541.gui CaMgO_full_541.gui
mv CaMgO_full_tmp_542.f9 CaMgO_full_542.f9
mv CaMgO_full_tmp_542.gui CaMgO_full_542.gui
mv CaMgO_full_tmp_543.f9 CaMgO_full_543.f9
mv CaMgO_full_tmp_543.gui CaMgO_full_543.gui
mv CaMgO_full_tmp_544.f9 CaMgO_full_544.f9
mv CaMgO_full_tmp_544.gui CaMgO_full_544.gui
mv CaMgO_full_tmp_545.f9 CaMgO_full_545.f9
mv CaMgO_full_tmp_545.gui CaMgO_full_545.gui
mv CaMgO_full_tmp_546.f9 CaMgO_full_546.f9
mv CaMgO_full_tmp_546.gui CaMgO_full_546.gui
mv CaMgO_full_tmp_547.f9 CaMgO_full_547.f9
mv CaMgO_full_tmp_547.gui CaMgO_full_547.gui
mv CaMgO_full_tmp_548.f9 CaMgO_full_548.f9
mv CaMgO_full_tmp_548.gui CaMgO_full_548.gui
mv CaMgO_full_tmp_549.f9 CaMgO_full_549.f9
mv CaMgO_full_tmp_549.gui CaMgO_full_549.gui
mv CaMgO_full_tmp_550.f9 CaMgO_full_550.f9
mv CaMgO_full_tmp_550.gui CaMgO_full_550.gui
mv CaMgO_full_tmp_551.f9 CaMgO_full_551.f9
mv CaMgO_full_tmp_551.gui CaMgO_full_551.gui
mv CaMgO_full_tmp_552.f9 CaMgO_full_552.f9
mv CaMgO_full_tmp_552.gui CaMgO_full_552.gui
mv CaMgO_full_tmp_553.f9 CaMgO_full_553.f9
mv CaMgO_full_tmp_553.gui CaMgO_full_553.gui
mv CaMgO_full_tmp_554.f9 CaMgO_full_554.f9
mv CaMgO_full_tmp_554.gui CaMgO_full_554.gui
mv CaMgO_full_tmp_555.f9 CaMgO_full_555.f9
mv CaMgO_full_tmp_555.gui CaMgO_full_555.gui
mv CaMgO_full_tmp_556.f9 CaMgO_full_556.f9
mv CaMgO_full_tmp_556.gui CaMgO_full_556.gui
mv CaMgO_full_tmp_557.f9 CaMgO_full_557.f9
mv CaMgO_full_tmp_557.gui CaMgO_full_557.gui
mv CaMgO_full_tmp_558.f9 CaMgO_full_558.f9
mv CaMgO_full_tmp_558.gui CaMgO_full_558.gui
mv CaMgO_full_tmp_559.f9 CaMgO_full_559.f9
mv CaMgO_full_tmp_559.gui CaMgO_full_559.gui
mv CaMgO_full_tmp_560.f9 CaMgO_full_560.f9
mv CaMgO_full_tmp_560.gui CaMgO_full_560.gui
mv CaMgO_full_tmp_561.f9 CaMgO_full_561.f9
mv CaMgO_full_tmp_561.gui CaMgO_full_561.gui
mv CaMgO_full_tmp_562.f9 CaMgO_full_562.f9
mv CaMgO_full_tmp_562.gui CaMgO_full_562.gui
mv CaMgO_full_tmp_563.f9 CaMgO_full_563.f9
mv CaMgO_full_tmp_563.gui CaMgO_full_563.gui
mv CaMgO_full_tmp_564.f9 CaMgO_full_564.f9
mv CaMgO_full_tmp_564.gui CaMgO_full_564.gui
mv CaMgO_full_tmp_565.f9 CaMgO_full_565.f9
mv CaMgO_full_tmp_565.gui CaMgO_full_565.gui
mv CaMgO_full_tmp_566.f9 CaMgO_full_566.f9
mv CaMgO_full_tmp_566.gui CaMgO_full_566.gui
mv CaMgO_full_tmp_567.f9 CaMgO_full_567.f9
mv CaMgO_full_tmp_567.gui CaMgO_full_567.gui
mv CaMgO_full_tmp_568.f9 CaMgO_full_568.f9
mv CaMgO_full_tmp_568.gui CaMgO_full_568.gui
mv CaMgO_full_tmp_569.f9 CaMgO_full_569.f9
mv CaMgO_full_tmp_569.gui CaMgO_full_569.gui
mv CaMgO_full_tmp_570.f9 CaMgO_full_570.f9
mv CaMgO_full_tmp_570.gui CaMgO_full_570.gui
mv CaMgO_full_tmp_571.f9 CaMgO_full_571.f9
mv CaMgO_full_tmp_571.gui CaMgO_full_571.gui
mv CaMgO_full_tmp_572.f9 CaMgO_full_572.f9
mv CaMgO_full_tmp_572.gui CaMgO_full_572.gui
mv CaMgO_full_tmp_573.f9 CaMgO_full_573.f9
mv CaMgO_full_tmp_573.gui CaMgO_full_573.gui
mv CaMgO_full_tmp_574.f9 CaMgO_full_574.f9
mv CaMgO_full_tmp_574.gui CaMgO_full_574.gui
mv CaMgO_full_tmp_575.f9 CaMgO_full_575.f9
mv CaMgO_full_tmp_575.gui CaMgO_full_575.gui
mv CaMgO_full_tmp_576.f9 CaMgO_full_576.f9
mv CaMgO_full_tmp_576.gui CaMgO_full_576.gui
mv CaMgO_full_tmp_577.f9 CaMgO_full_577.f9
mv CaMgO_full_tmp_577.gui CaMgO_full_577.gui
mv CaMgO_full_tmp_578.f9 CaMgO_full_578.f9
mv CaMgO_full_tmp_578.gui CaMgO_full_578.gui
mv CaMgO_full_tmp_579.f9 CaMgO_full_579.f9
mv CaMgO_full_tmp_579.gui CaMgO_full_579.gui
mv CaMgO_full_tmp_580.f9 CaMgO_full_580.f9
mv CaMgO_full_tmp_580.gui CaMgO_full_580.gui
mv CaMgO_full_tmp_581.f9 CaMgO_full_581.f9
mv CaMgO_full_tmp_581.gui CaMgO_full_581.gui
mv CaMgO_full_tmp_582.f9 CaMgO_full_582.f9
mv CaMgO_full_tmp_582.gui CaMgO_full_582.gui
mv CaMgO_full_tmp_583.f9 CaMgO_full_583.f9
mv CaMgO_full_tmp_583.gui CaMgO_full_583.gui
mv CaMgO_full_tmp_584.f9 CaMgO_full_584.f9
mv CaMgO_full_tmp_584.gui CaMgO_full_584.gui
mv CaMgO_full_tmp_585.f9 CaMgO_full_585.f9
mv CaMgO_full_tmp_585.gui CaMgO_full_585.gui
mv CaMgO_full_tmp_586.f9 CaMgO_full_586.f9
mv CaMgO_full_tmp_586.gui CaMgO_full_586.gui
mv CaMgO_full_tmp_587.f9 CaMgO_full_587.f9
mv CaMgO_full_tmp_587.gui CaMgO_full_587.gui
mv CaMgO_full_tmp_588.f9 CaMgO_full_588.f9
mv CaMgO_full_tmp_588.gui CaMgO_full_588.gui
mv CaMgO_full_tmp_589.f9 CaMgO_full_589.f9
mv CaMgO_full_tmp_589.gui CaMgO_full_589.gui
mv CaMgO_full_tmp_590.f9 CaMgO_full_590.f9
mv CaMgO_full_tmp_590.gui CaMgO_full_590.gui
mv CaMgO_full_tmp_591.f9 CaMgO_full_591.f9
mv CaMgO_full_tmp_591.gui CaMgO_full_591.gui
mv CaMgO_full_tmp_592.f9 CaMgO_full_592.f9
mv CaMgO_full_tmp_592.gui CaMgO_full_592.gui
mv CaMgO_full_tmp_593.f9 CaMgO_full_593.f9
mv CaMgO_full_tmp_593.gui CaMgO_full_593.gui
mv CaMgO_full_tmp_594.f9 CaMgO_full_594.f9
mv CaMgO_full_tmp_594.gui CaMgO_full_594.gui
mv CaMgO_full_tmp_595.f9 CaMgO_full_595.f9
mv CaMgO_full_tmp_595.gui CaMgO_full_595.gui
mv CaMgO_full_tmp_596.f9 CaMgO_full_596.f9
mv CaMgO_full_tmp_596.gui CaMgO_full_596.gui
mv CaMgO_full_tmp_597.f9 CaMgO_full_597.f9
mv CaMgO_full_tmp_597.gui CaMgO_full_597.gui
mv CaMgO_full_tmp_598.f9 CaMgO_full_598.f9
mv CaMgO_full_tmp_598.gui CaMgO_full_598.gui
mv CaMgO_full_tmp_599.f9 CaMgO_full_599.f9
mv CaMgO_full_tmp_599.gui CaMgO_full_599.gui
