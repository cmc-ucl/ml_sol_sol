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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_501 CaMgO_coord_501
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_501


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_502 CaMgO_coord_502
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_502


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_503 CaMgO_coord_503
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_503


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_504 CaMgO_coord_504
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_504


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_505 CaMgO_coord_505
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_505


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_506 CaMgO_coord_506
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_506


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_507 CaMgO_coord_507
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_507


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_508 CaMgO_coord_508
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_508


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_509 CaMgO_coord_509
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_509


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_510 CaMgO_coord_510
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_510


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_511 CaMgO_coord_511
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_511


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_512 CaMgO_coord_512
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_512


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_513 CaMgO_coord_513
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_513


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_514 CaMgO_coord_514
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_514


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_515 CaMgO_coord_515
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_515


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_516 CaMgO_coord_516
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_516


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_517 CaMgO_coord_517
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_517


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_518 CaMgO_coord_518
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_518


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_519 CaMgO_coord_519
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_519


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_520 CaMgO_coord_520
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_520


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_521 CaMgO_coord_521
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_521


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_522 CaMgO_coord_522
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_522


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_523 CaMgO_coord_523
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_523


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_524 CaMgO_coord_524
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_524


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_525 CaMgO_coord_525
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_525


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_526 CaMgO_coord_526
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_526


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_527 CaMgO_coord_527
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_527


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_528 CaMgO_coord_528
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_528


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_529 CaMgO_coord_529
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_529


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_530 CaMgO_coord_530
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_530


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_531 CaMgO_coord_531
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_531


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_532 CaMgO_coord_532
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_532


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_533 CaMgO_coord_533
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_533


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_534 CaMgO_coord_534
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_534


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_535 CaMgO_coord_535
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_535


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_536 CaMgO_coord_536
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_536


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_537 CaMgO_coord_537
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_537


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_538 CaMgO_coord_538
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_538


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_539 CaMgO_coord_539
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_539


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_540 CaMgO_coord_540
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_540


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_541 CaMgO_coord_541
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_541


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_542 CaMgO_coord_542
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_542


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_543 CaMgO_coord_543
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_543


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_544 CaMgO_coord_544
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_544


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_545 CaMgO_coord_545
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_545


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_546 CaMgO_coord_546
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_546


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_547 CaMgO_coord_547
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_547


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_548 CaMgO_coord_548
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_548


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_549 CaMgO_coord_549
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_549


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_550 CaMgO_coord_550
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_550


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_551 CaMgO_coord_551
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_551


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_552 CaMgO_coord_552
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_552


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_553 CaMgO_coord_553
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_553


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_554 CaMgO_coord_554
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_554


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_555 CaMgO_coord_555
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_555


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_556 CaMgO_coord_556
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_556


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_557 CaMgO_coord_557
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_557


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_558 CaMgO_coord_558
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_558


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_559 CaMgO_coord_559
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_559


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_560 CaMgO_coord_560
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_560


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_561 CaMgO_coord_561
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_561


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_562 CaMgO_coord_562
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_562


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_563 CaMgO_coord_563
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_563


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_564 CaMgO_coord_564
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_564


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_565 CaMgO_coord_565
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_565


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_566 CaMgO_coord_566
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_566


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_567 CaMgO_coord_567
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_567


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_568 CaMgO_coord_568
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_568


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_569 CaMgO_coord_569
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_569


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_570 CaMgO_coord_570
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_570


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_571 CaMgO_coord_571
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_571


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_572 CaMgO_coord_572
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_572


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_573 CaMgO_coord_573
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_573


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_574 CaMgO_coord_574
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_574


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_575 CaMgO_coord_575
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_575


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_576 CaMgO_coord_576
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_576


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_577 CaMgO_coord_577
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_577


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_578 CaMgO_coord_578
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_578


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_579 CaMgO_coord_579
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_579


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_580 CaMgO_coord_580
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_580


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_581 CaMgO_coord_581
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_581


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_582 CaMgO_coord_582
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_582


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_583 CaMgO_coord_583
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_583


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_584 CaMgO_coord_584
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_584


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_585 CaMgO_coord_585
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_585


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_586 CaMgO_coord_586
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_586


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_587 CaMgO_coord_587
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_587


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_588 CaMgO_coord_588
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_588


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_589 CaMgO_coord_589
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_589


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_590 CaMgO_coord_590
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_590


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_591 CaMgO_coord_591
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_591


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_592 CaMgO_coord_592
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_592


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_593 CaMgO_coord_593
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_593


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_594 CaMgO_coord_594
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_594


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_595 CaMgO_coord_595
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_595


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_596 CaMgO_coord_596
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_596


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_597 CaMgO_coord_597
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_597


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_598 CaMgO_coord_598
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_598


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_599 CaMgO_coord_599
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_599


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_600 CaMgO_coord_600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_600


