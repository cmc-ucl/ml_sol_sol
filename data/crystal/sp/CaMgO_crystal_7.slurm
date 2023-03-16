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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_699


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_700


