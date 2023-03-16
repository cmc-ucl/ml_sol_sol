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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_600 CaMgO_full_tmp_600 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_601 CaMgO_full_tmp_601 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_602 CaMgO_full_tmp_602 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_603 CaMgO_full_tmp_603 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_604 CaMgO_full_tmp_604 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_605 CaMgO_full_tmp_605 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_606 CaMgO_full_tmp_606 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_607 CaMgO_full_tmp_607 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_608 CaMgO_full_tmp_608 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_609 CaMgO_full_tmp_609 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_610 CaMgO_full_tmp_610 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_611 CaMgO_full_tmp_611 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_612 CaMgO_full_tmp_612 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_613 CaMgO_full_tmp_613 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_614 CaMgO_full_tmp_614 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_615 CaMgO_full_tmp_615 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_616 CaMgO_full_tmp_616 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_617 CaMgO_full_tmp_617 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_618 CaMgO_full_tmp_618 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_619 CaMgO_full_tmp_619 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_620 CaMgO_full_tmp_620 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_621 CaMgO_full_tmp_621 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_622 CaMgO_full_tmp_622 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_623 CaMgO_full_tmp_623 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_624 CaMgO_full_tmp_624 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_625 CaMgO_full_tmp_625 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_626 CaMgO_full_tmp_626 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_627 CaMgO_full_tmp_627 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_628 CaMgO_full_tmp_628 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_629 CaMgO_full_tmp_629 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_630 CaMgO_full_tmp_630 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_631 CaMgO_full_tmp_631 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_632 CaMgO_full_tmp_632 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_633 CaMgO_full_tmp_633 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_634 CaMgO_full_tmp_634 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_635 CaMgO_full_tmp_635 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_636 CaMgO_full_tmp_636 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_637 CaMgO_full_tmp_637 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_638 CaMgO_full_tmp_638 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_639 CaMgO_full_tmp_639 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_640 CaMgO_full_tmp_640 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_641 CaMgO_full_tmp_641 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_642 CaMgO_full_tmp_642 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_643 CaMgO_full_tmp_643 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_644 CaMgO_full_tmp_644 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_645 CaMgO_full_tmp_645 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_646 CaMgO_full_tmp_646 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_647 CaMgO_full_tmp_647 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_648 CaMgO_full_tmp_648 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_649 CaMgO_full_tmp_649 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_650 CaMgO_full_tmp_650 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_651 CaMgO_full_tmp_651 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_652 CaMgO_full_tmp_652 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_653 CaMgO_full_tmp_653 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_654 CaMgO_full_tmp_654 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_655 CaMgO_full_tmp_655 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_656 CaMgO_full_tmp_656 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_657 CaMgO_full_tmp_657 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_658 CaMgO_full_tmp_658 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_659 CaMgO_full_tmp_659 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_660 CaMgO_full_tmp_660 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_661 CaMgO_full_tmp_661 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_662 CaMgO_full_tmp_662 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_663 CaMgO_full_tmp_663 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_664 CaMgO_full_tmp_664 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_665 CaMgO_full_tmp_665 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_666 CaMgO_full_tmp_666 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_667 CaMgO_full_tmp_667 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_668 CaMgO_full_tmp_668 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_669 CaMgO_full_tmp_669 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_670 CaMgO_full_tmp_670 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_671 CaMgO_full_tmp_671 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_672 CaMgO_full_tmp_672 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_673 CaMgO_full_tmp_673 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_674 CaMgO_full_tmp_674 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_675 CaMgO_full_tmp_675 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_676 CaMgO_full_tmp_676 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_677 CaMgO_full_tmp_677 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_678 CaMgO_full_tmp_678 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_679 CaMgO_full_tmp_679 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_680 CaMgO_full_tmp_680 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_681 CaMgO_full_tmp_681 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_682 CaMgO_full_tmp_682 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_683 CaMgO_full_tmp_683 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_684 CaMgO_full_tmp_684 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_685 CaMgO_full_tmp_685 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_686 CaMgO_full_tmp_686 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_687 CaMgO_full_tmp_687 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_688 CaMgO_full_tmp_688 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_689 CaMgO_full_tmp_689 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_690 CaMgO_full_tmp_690 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_691 CaMgO_full_tmp_691 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_692 CaMgO_full_tmp_692 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_693 CaMgO_full_tmp_693 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_694 CaMgO_full_tmp_694 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_695 CaMgO_full_tmp_695 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_696 CaMgO_full_tmp_696 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_697 CaMgO_full_tmp_697 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_698 CaMgO_full_tmp_698 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_699 CaMgO_full_tmp_699 &

wait
mv CaMgO_full_tmp_600.f9 CaMgO_full_600.f9
mv CaMgO_full_tmp_600.gui CaMgO_full_600.gui
mv CaMgO_full_tmp_601.f9 CaMgO_full_601.f9
mv CaMgO_full_tmp_601.gui CaMgO_full_601.gui
mv CaMgO_full_tmp_602.f9 CaMgO_full_602.f9
mv CaMgO_full_tmp_602.gui CaMgO_full_602.gui
mv CaMgO_full_tmp_603.f9 CaMgO_full_603.f9
mv CaMgO_full_tmp_603.gui CaMgO_full_603.gui
mv CaMgO_full_tmp_604.f9 CaMgO_full_604.f9
mv CaMgO_full_tmp_604.gui CaMgO_full_604.gui
mv CaMgO_full_tmp_605.f9 CaMgO_full_605.f9
mv CaMgO_full_tmp_605.gui CaMgO_full_605.gui
mv CaMgO_full_tmp_606.f9 CaMgO_full_606.f9
mv CaMgO_full_tmp_606.gui CaMgO_full_606.gui
mv CaMgO_full_tmp_607.f9 CaMgO_full_607.f9
mv CaMgO_full_tmp_607.gui CaMgO_full_607.gui
mv CaMgO_full_tmp_608.f9 CaMgO_full_608.f9
mv CaMgO_full_tmp_608.gui CaMgO_full_608.gui
mv CaMgO_full_tmp_609.f9 CaMgO_full_609.f9
mv CaMgO_full_tmp_609.gui CaMgO_full_609.gui
mv CaMgO_full_tmp_610.f9 CaMgO_full_610.f9
mv CaMgO_full_tmp_610.gui CaMgO_full_610.gui
mv CaMgO_full_tmp_611.f9 CaMgO_full_611.f9
mv CaMgO_full_tmp_611.gui CaMgO_full_611.gui
mv CaMgO_full_tmp_612.f9 CaMgO_full_612.f9
mv CaMgO_full_tmp_612.gui CaMgO_full_612.gui
mv CaMgO_full_tmp_613.f9 CaMgO_full_613.f9
mv CaMgO_full_tmp_613.gui CaMgO_full_613.gui
mv CaMgO_full_tmp_614.f9 CaMgO_full_614.f9
mv CaMgO_full_tmp_614.gui CaMgO_full_614.gui
mv CaMgO_full_tmp_615.f9 CaMgO_full_615.f9
mv CaMgO_full_tmp_615.gui CaMgO_full_615.gui
mv CaMgO_full_tmp_616.f9 CaMgO_full_616.f9
mv CaMgO_full_tmp_616.gui CaMgO_full_616.gui
mv CaMgO_full_tmp_617.f9 CaMgO_full_617.f9
mv CaMgO_full_tmp_617.gui CaMgO_full_617.gui
mv CaMgO_full_tmp_618.f9 CaMgO_full_618.f9
mv CaMgO_full_tmp_618.gui CaMgO_full_618.gui
mv CaMgO_full_tmp_619.f9 CaMgO_full_619.f9
mv CaMgO_full_tmp_619.gui CaMgO_full_619.gui
mv CaMgO_full_tmp_620.f9 CaMgO_full_620.f9
mv CaMgO_full_tmp_620.gui CaMgO_full_620.gui
mv CaMgO_full_tmp_621.f9 CaMgO_full_621.f9
mv CaMgO_full_tmp_621.gui CaMgO_full_621.gui
mv CaMgO_full_tmp_622.f9 CaMgO_full_622.f9
mv CaMgO_full_tmp_622.gui CaMgO_full_622.gui
mv CaMgO_full_tmp_623.f9 CaMgO_full_623.f9
mv CaMgO_full_tmp_623.gui CaMgO_full_623.gui
mv CaMgO_full_tmp_624.f9 CaMgO_full_624.f9
mv CaMgO_full_tmp_624.gui CaMgO_full_624.gui
mv CaMgO_full_tmp_625.f9 CaMgO_full_625.f9
mv CaMgO_full_tmp_625.gui CaMgO_full_625.gui
mv CaMgO_full_tmp_626.f9 CaMgO_full_626.f9
mv CaMgO_full_tmp_626.gui CaMgO_full_626.gui
mv CaMgO_full_tmp_627.f9 CaMgO_full_627.f9
mv CaMgO_full_tmp_627.gui CaMgO_full_627.gui
mv CaMgO_full_tmp_628.f9 CaMgO_full_628.f9
mv CaMgO_full_tmp_628.gui CaMgO_full_628.gui
mv CaMgO_full_tmp_629.f9 CaMgO_full_629.f9
mv CaMgO_full_tmp_629.gui CaMgO_full_629.gui
mv CaMgO_full_tmp_630.f9 CaMgO_full_630.f9
mv CaMgO_full_tmp_630.gui CaMgO_full_630.gui
mv CaMgO_full_tmp_631.f9 CaMgO_full_631.f9
mv CaMgO_full_tmp_631.gui CaMgO_full_631.gui
mv CaMgO_full_tmp_632.f9 CaMgO_full_632.f9
mv CaMgO_full_tmp_632.gui CaMgO_full_632.gui
mv CaMgO_full_tmp_633.f9 CaMgO_full_633.f9
mv CaMgO_full_tmp_633.gui CaMgO_full_633.gui
mv CaMgO_full_tmp_634.f9 CaMgO_full_634.f9
mv CaMgO_full_tmp_634.gui CaMgO_full_634.gui
mv CaMgO_full_tmp_635.f9 CaMgO_full_635.f9
mv CaMgO_full_tmp_635.gui CaMgO_full_635.gui
mv CaMgO_full_tmp_636.f9 CaMgO_full_636.f9
mv CaMgO_full_tmp_636.gui CaMgO_full_636.gui
mv CaMgO_full_tmp_637.f9 CaMgO_full_637.f9
mv CaMgO_full_tmp_637.gui CaMgO_full_637.gui
mv CaMgO_full_tmp_638.f9 CaMgO_full_638.f9
mv CaMgO_full_tmp_638.gui CaMgO_full_638.gui
mv CaMgO_full_tmp_639.f9 CaMgO_full_639.f9
mv CaMgO_full_tmp_639.gui CaMgO_full_639.gui
mv CaMgO_full_tmp_640.f9 CaMgO_full_640.f9
mv CaMgO_full_tmp_640.gui CaMgO_full_640.gui
mv CaMgO_full_tmp_641.f9 CaMgO_full_641.f9
mv CaMgO_full_tmp_641.gui CaMgO_full_641.gui
mv CaMgO_full_tmp_642.f9 CaMgO_full_642.f9
mv CaMgO_full_tmp_642.gui CaMgO_full_642.gui
mv CaMgO_full_tmp_643.f9 CaMgO_full_643.f9
mv CaMgO_full_tmp_643.gui CaMgO_full_643.gui
mv CaMgO_full_tmp_644.f9 CaMgO_full_644.f9
mv CaMgO_full_tmp_644.gui CaMgO_full_644.gui
mv CaMgO_full_tmp_645.f9 CaMgO_full_645.f9
mv CaMgO_full_tmp_645.gui CaMgO_full_645.gui
mv CaMgO_full_tmp_646.f9 CaMgO_full_646.f9
mv CaMgO_full_tmp_646.gui CaMgO_full_646.gui
mv CaMgO_full_tmp_647.f9 CaMgO_full_647.f9
mv CaMgO_full_tmp_647.gui CaMgO_full_647.gui
mv CaMgO_full_tmp_648.f9 CaMgO_full_648.f9
mv CaMgO_full_tmp_648.gui CaMgO_full_648.gui
mv CaMgO_full_tmp_649.f9 CaMgO_full_649.f9
mv CaMgO_full_tmp_649.gui CaMgO_full_649.gui
mv CaMgO_full_tmp_650.f9 CaMgO_full_650.f9
mv CaMgO_full_tmp_650.gui CaMgO_full_650.gui
mv CaMgO_full_tmp_651.f9 CaMgO_full_651.f9
mv CaMgO_full_tmp_651.gui CaMgO_full_651.gui
mv CaMgO_full_tmp_652.f9 CaMgO_full_652.f9
mv CaMgO_full_tmp_652.gui CaMgO_full_652.gui
mv CaMgO_full_tmp_653.f9 CaMgO_full_653.f9
mv CaMgO_full_tmp_653.gui CaMgO_full_653.gui
mv CaMgO_full_tmp_654.f9 CaMgO_full_654.f9
mv CaMgO_full_tmp_654.gui CaMgO_full_654.gui
mv CaMgO_full_tmp_655.f9 CaMgO_full_655.f9
mv CaMgO_full_tmp_655.gui CaMgO_full_655.gui
mv CaMgO_full_tmp_656.f9 CaMgO_full_656.f9
mv CaMgO_full_tmp_656.gui CaMgO_full_656.gui
mv CaMgO_full_tmp_657.f9 CaMgO_full_657.f9
mv CaMgO_full_tmp_657.gui CaMgO_full_657.gui
mv CaMgO_full_tmp_658.f9 CaMgO_full_658.f9
mv CaMgO_full_tmp_658.gui CaMgO_full_658.gui
mv CaMgO_full_tmp_659.f9 CaMgO_full_659.f9
mv CaMgO_full_tmp_659.gui CaMgO_full_659.gui
mv CaMgO_full_tmp_660.f9 CaMgO_full_660.f9
mv CaMgO_full_tmp_660.gui CaMgO_full_660.gui
mv CaMgO_full_tmp_661.f9 CaMgO_full_661.f9
mv CaMgO_full_tmp_661.gui CaMgO_full_661.gui
mv CaMgO_full_tmp_662.f9 CaMgO_full_662.f9
mv CaMgO_full_tmp_662.gui CaMgO_full_662.gui
mv CaMgO_full_tmp_663.f9 CaMgO_full_663.f9
mv CaMgO_full_tmp_663.gui CaMgO_full_663.gui
mv CaMgO_full_tmp_664.f9 CaMgO_full_664.f9
mv CaMgO_full_tmp_664.gui CaMgO_full_664.gui
mv CaMgO_full_tmp_665.f9 CaMgO_full_665.f9
mv CaMgO_full_tmp_665.gui CaMgO_full_665.gui
mv CaMgO_full_tmp_666.f9 CaMgO_full_666.f9
mv CaMgO_full_tmp_666.gui CaMgO_full_666.gui
mv CaMgO_full_tmp_667.f9 CaMgO_full_667.f9
mv CaMgO_full_tmp_667.gui CaMgO_full_667.gui
mv CaMgO_full_tmp_668.f9 CaMgO_full_668.f9
mv CaMgO_full_tmp_668.gui CaMgO_full_668.gui
mv CaMgO_full_tmp_669.f9 CaMgO_full_669.f9
mv CaMgO_full_tmp_669.gui CaMgO_full_669.gui
mv CaMgO_full_tmp_670.f9 CaMgO_full_670.f9
mv CaMgO_full_tmp_670.gui CaMgO_full_670.gui
mv CaMgO_full_tmp_671.f9 CaMgO_full_671.f9
mv CaMgO_full_tmp_671.gui CaMgO_full_671.gui
mv CaMgO_full_tmp_672.f9 CaMgO_full_672.f9
mv CaMgO_full_tmp_672.gui CaMgO_full_672.gui
mv CaMgO_full_tmp_673.f9 CaMgO_full_673.f9
mv CaMgO_full_tmp_673.gui CaMgO_full_673.gui
mv CaMgO_full_tmp_674.f9 CaMgO_full_674.f9
mv CaMgO_full_tmp_674.gui CaMgO_full_674.gui
mv CaMgO_full_tmp_675.f9 CaMgO_full_675.f9
mv CaMgO_full_tmp_675.gui CaMgO_full_675.gui
mv CaMgO_full_tmp_676.f9 CaMgO_full_676.f9
mv CaMgO_full_tmp_676.gui CaMgO_full_676.gui
mv CaMgO_full_tmp_677.f9 CaMgO_full_677.f9
mv CaMgO_full_tmp_677.gui CaMgO_full_677.gui
mv CaMgO_full_tmp_678.f9 CaMgO_full_678.f9
mv CaMgO_full_tmp_678.gui CaMgO_full_678.gui
mv CaMgO_full_tmp_679.f9 CaMgO_full_679.f9
mv CaMgO_full_tmp_679.gui CaMgO_full_679.gui
mv CaMgO_full_tmp_680.f9 CaMgO_full_680.f9
mv CaMgO_full_tmp_680.gui CaMgO_full_680.gui
mv CaMgO_full_tmp_681.f9 CaMgO_full_681.f9
mv CaMgO_full_tmp_681.gui CaMgO_full_681.gui
mv CaMgO_full_tmp_682.f9 CaMgO_full_682.f9
mv CaMgO_full_tmp_682.gui CaMgO_full_682.gui
mv CaMgO_full_tmp_683.f9 CaMgO_full_683.f9
mv CaMgO_full_tmp_683.gui CaMgO_full_683.gui
mv CaMgO_full_tmp_684.f9 CaMgO_full_684.f9
mv CaMgO_full_tmp_684.gui CaMgO_full_684.gui
mv CaMgO_full_tmp_685.f9 CaMgO_full_685.f9
mv CaMgO_full_tmp_685.gui CaMgO_full_685.gui
mv CaMgO_full_tmp_686.f9 CaMgO_full_686.f9
mv CaMgO_full_tmp_686.gui CaMgO_full_686.gui
mv CaMgO_full_tmp_687.f9 CaMgO_full_687.f9
mv CaMgO_full_tmp_687.gui CaMgO_full_687.gui
mv CaMgO_full_tmp_688.f9 CaMgO_full_688.f9
mv CaMgO_full_tmp_688.gui CaMgO_full_688.gui
mv CaMgO_full_tmp_689.f9 CaMgO_full_689.f9
mv CaMgO_full_tmp_689.gui CaMgO_full_689.gui
mv CaMgO_full_tmp_690.f9 CaMgO_full_690.f9
mv CaMgO_full_tmp_690.gui CaMgO_full_690.gui
mv CaMgO_full_tmp_691.f9 CaMgO_full_691.f9
mv CaMgO_full_tmp_691.gui CaMgO_full_691.gui
mv CaMgO_full_tmp_692.f9 CaMgO_full_692.f9
mv CaMgO_full_tmp_692.gui CaMgO_full_692.gui
mv CaMgO_full_tmp_693.f9 CaMgO_full_693.f9
mv CaMgO_full_tmp_693.gui CaMgO_full_693.gui
mv CaMgO_full_tmp_694.f9 CaMgO_full_694.f9
mv CaMgO_full_tmp_694.gui CaMgO_full_694.gui
mv CaMgO_full_tmp_695.f9 CaMgO_full_695.f9
mv CaMgO_full_tmp_695.gui CaMgO_full_695.gui
mv CaMgO_full_tmp_696.f9 CaMgO_full_696.f9
mv CaMgO_full_tmp_696.gui CaMgO_full_696.gui
mv CaMgO_full_tmp_697.f9 CaMgO_full_697.f9
mv CaMgO_full_tmp_697.gui CaMgO_full_697.gui
mv CaMgO_full_tmp_698.f9 CaMgO_full_698.f9
mv CaMgO_full_tmp_698.gui CaMgO_full_698.gui
mv CaMgO_full_tmp_699.f9 CaMgO_full_699.f9
mv CaMgO_full_tmp_699.gui CaMgO_full_699.gui
