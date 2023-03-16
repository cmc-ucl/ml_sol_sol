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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_501
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_501


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_502
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_502


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_503
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_503


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_504
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_504


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_505
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_505


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_506
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_506


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_507
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_507


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_508
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_508


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_509
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_509


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_510
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_510


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_511
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_511


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_512
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_512


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_513
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_513


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_514
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_514


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_515
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_515


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_516
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_516


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_517
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_517


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_518
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_518


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_519
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_519


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_520
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_520


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_521
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_521


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_522
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_522


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_523
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_523


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_524
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_524


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_525
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_525


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_526
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_526


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_527
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_527


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_528
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_528


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_529
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_529


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_530
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_530


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_531
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_531


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_532
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_532


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_533
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_533


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_534
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_534


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_535
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_535


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_536
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_536


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_537
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_537


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_538
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_538


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_539
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_539


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_540
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_540


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_541
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_541


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_542
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_542


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_543
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_543


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_544
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_544


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_545
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_545


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_546
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_546


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_547
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_547


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_548
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_548


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_549
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_549


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_550
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_550


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_551
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_551


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_552
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_552


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_553
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_553


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_554
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_554


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_555
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_555


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_556
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_556


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_557
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_557


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_558
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_558


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_559
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_559


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_560
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_560


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_561
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_561


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_562
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_562


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_563
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_563


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_564
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_564


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_565
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_565


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_566
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_566


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_567
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_567


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_568
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_568


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_569
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_569


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_570
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_570


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_571
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_571


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_572
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_572


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_573
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_573


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_574
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_574


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_575
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_575


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_576
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_576


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_577
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_577


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_578
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_578


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_579
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_579


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_580
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_580


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_581
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_581


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_582
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_582


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_583
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_583


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_584
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_584


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_585
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_585


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_586
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_586


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_587
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_587


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_588
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_588


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_589
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_589


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_590
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_590


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_591
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_591


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_592
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_592


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_593
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_593


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_594
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_594


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_595
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_595


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_596
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_596


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_597
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_597


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_598
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_598


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_599
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_599


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_600


