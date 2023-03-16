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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_601 CaMgO_coord_601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_602 CaMgO_coord_602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_603 CaMgO_coord_603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_604 CaMgO_coord_604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_605 CaMgO_coord_605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_606 CaMgO_coord_606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_607 CaMgO_coord_607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_608 CaMgO_coord_608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_609 CaMgO_coord_609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_610 CaMgO_coord_610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_611 CaMgO_coord_611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_612 CaMgO_coord_612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_613 CaMgO_coord_613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_614 CaMgO_coord_614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_615 CaMgO_coord_615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_616 CaMgO_coord_616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_617 CaMgO_coord_617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_618 CaMgO_coord_618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_619 CaMgO_coord_619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_620 CaMgO_coord_620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_621 CaMgO_coord_621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_622 CaMgO_coord_622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_623 CaMgO_coord_623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_624 CaMgO_coord_624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_625 CaMgO_coord_625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_626 CaMgO_coord_626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_627 CaMgO_coord_627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_628 CaMgO_coord_628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_629 CaMgO_coord_629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_630 CaMgO_coord_630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_631 CaMgO_coord_631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_632 CaMgO_coord_632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_633 CaMgO_coord_633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_634 CaMgO_coord_634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_635 CaMgO_coord_635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_636 CaMgO_coord_636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_637 CaMgO_coord_637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_638 CaMgO_coord_638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_639 CaMgO_coord_639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_640 CaMgO_coord_640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_641 CaMgO_coord_641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_642 CaMgO_coord_642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_643 CaMgO_coord_643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_644 CaMgO_coord_644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_645 CaMgO_coord_645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_646 CaMgO_coord_646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_647 CaMgO_coord_647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_648 CaMgO_coord_648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_649 CaMgO_coord_649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_650 CaMgO_coord_650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_651 CaMgO_coord_651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_652 CaMgO_coord_652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_653 CaMgO_coord_653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_654 CaMgO_coord_654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_655 CaMgO_coord_655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_656 CaMgO_coord_656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_657 CaMgO_coord_657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_658 CaMgO_coord_658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_659 CaMgO_coord_659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_660 CaMgO_coord_660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_661 CaMgO_coord_661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_662 CaMgO_coord_662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_663 CaMgO_coord_663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_664 CaMgO_coord_664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_665 CaMgO_coord_665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_666 CaMgO_coord_666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_667 CaMgO_coord_667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_668 CaMgO_coord_668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_669 CaMgO_coord_669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_670 CaMgO_coord_670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_671 CaMgO_coord_671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_672 CaMgO_coord_672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_673 CaMgO_coord_673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_674 CaMgO_coord_674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_675 CaMgO_coord_675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_676 CaMgO_coord_676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_677 CaMgO_coord_677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_678 CaMgO_coord_678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_679 CaMgO_coord_679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_680 CaMgO_coord_680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_681 CaMgO_coord_681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_682 CaMgO_coord_682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_683 CaMgO_coord_683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_684 CaMgO_coord_684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_685 CaMgO_coord_685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_686 CaMgO_coord_686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_687 CaMgO_coord_687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_688 CaMgO_coord_688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_689 CaMgO_coord_689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_690 CaMgO_coord_690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_691 CaMgO_coord_691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_692 CaMgO_coord_692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_693 CaMgO_coord_693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_694 CaMgO_coord_694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_695 CaMgO_coord_695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_696 CaMgO_coord_696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_697 CaMgO_coord_697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_698 CaMgO_coord_698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_699 CaMgO_coord_699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_699


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_700 CaMgO_coord_700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_700


