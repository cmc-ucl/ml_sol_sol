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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_700 CaMgO_full_tmp_700 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_701 CaMgO_full_tmp_701 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_702 CaMgO_full_tmp_702 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_703 CaMgO_full_tmp_703 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_704 CaMgO_full_tmp_704 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_705 CaMgO_full_tmp_705 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_706 CaMgO_full_tmp_706 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_707 CaMgO_full_tmp_707 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_708 CaMgO_full_tmp_708 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_709 CaMgO_full_tmp_709 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_710 CaMgO_full_tmp_710 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_711 CaMgO_full_tmp_711 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_712 CaMgO_full_tmp_712 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_713 CaMgO_full_tmp_713 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_714 CaMgO_full_tmp_714 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_715 CaMgO_full_tmp_715 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_716 CaMgO_full_tmp_716 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_717 CaMgO_full_tmp_717 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_718 CaMgO_full_tmp_718 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_719 CaMgO_full_tmp_719 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_720 CaMgO_full_tmp_720 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_721 CaMgO_full_tmp_721 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_722 CaMgO_full_tmp_722 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_723 CaMgO_full_tmp_723 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_724 CaMgO_full_tmp_724 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_725 CaMgO_full_tmp_725 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_726 CaMgO_full_tmp_726 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_727 CaMgO_full_tmp_727 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_728 CaMgO_full_tmp_728 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_729 CaMgO_full_tmp_729 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_730 CaMgO_full_tmp_730 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_731 CaMgO_full_tmp_731 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_732 CaMgO_full_tmp_732 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_733 CaMgO_full_tmp_733 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_734 CaMgO_full_tmp_734 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_735 CaMgO_full_tmp_735 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_736 CaMgO_full_tmp_736 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_737 CaMgO_full_tmp_737 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_738 CaMgO_full_tmp_738 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_739 CaMgO_full_tmp_739 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_740 CaMgO_full_tmp_740 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_741 CaMgO_full_tmp_741 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_742 CaMgO_full_tmp_742 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_743 CaMgO_full_tmp_743 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_744 CaMgO_full_tmp_744 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_745 CaMgO_full_tmp_745 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_746 CaMgO_full_tmp_746 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_747 CaMgO_full_tmp_747 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_748 CaMgO_full_tmp_748 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_749 CaMgO_full_tmp_749 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_750 CaMgO_full_tmp_750 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_751 CaMgO_full_tmp_751 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_752 CaMgO_full_tmp_752 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_753 CaMgO_full_tmp_753 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_754 CaMgO_full_tmp_754 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_755 CaMgO_full_tmp_755 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_756 CaMgO_full_tmp_756 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_757 CaMgO_full_tmp_757 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_758 CaMgO_full_tmp_758 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_759 CaMgO_full_tmp_759 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_760 CaMgO_full_tmp_760 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_761 CaMgO_full_tmp_761 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_762 CaMgO_full_tmp_762 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_763 CaMgO_full_tmp_763 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_764 CaMgO_full_tmp_764 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_765 CaMgO_full_tmp_765 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_766 CaMgO_full_tmp_766 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_767 CaMgO_full_tmp_767 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_768 CaMgO_full_tmp_768 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_769 CaMgO_full_tmp_769 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_770 CaMgO_full_tmp_770 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_771 CaMgO_full_tmp_771 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_772 CaMgO_full_tmp_772 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_773 CaMgO_full_tmp_773 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_774 CaMgO_full_tmp_774 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_775 CaMgO_full_tmp_775 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_776 CaMgO_full_tmp_776 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_777 CaMgO_full_tmp_777 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_778 CaMgO_full_tmp_778 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_779 CaMgO_full_tmp_779 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_780 CaMgO_full_tmp_780 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_781 CaMgO_full_tmp_781 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_782 CaMgO_full_tmp_782 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_783 CaMgO_full_tmp_783 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_784 CaMgO_full_tmp_784 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_785 CaMgO_full_tmp_785 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_786 CaMgO_full_tmp_786 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_787 CaMgO_full_tmp_787 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_788 CaMgO_full_tmp_788 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_789 CaMgO_full_tmp_789 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_790 CaMgO_full_tmp_790 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_791 CaMgO_full_tmp_791 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_792 CaMgO_full_tmp_792 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_793 CaMgO_full_tmp_793 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_794 CaMgO_full_tmp_794 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_795 CaMgO_full_tmp_795 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_796 CaMgO_full_tmp_796 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_797 CaMgO_full_tmp_797 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_798 CaMgO_full_tmp_798 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_799 CaMgO_full_tmp_799 &

wait
mv CaMgO_full_tmp_700.f9 CaMgO_full_700.f9
mv CaMgO_full_tmp_700.gui CaMgO_full_700.gui
mv CaMgO_full_tmp_701.f9 CaMgO_full_701.f9
mv CaMgO_full_tmp_701.gui CaMgO_full_701.gui
mv CaMgO_full_tmp_702.f9 CaMgO_full_702.f9
mv CaMgO_full_tmp_702.gui CaMgO_full_702.gui
mv CaMgO_full_tmp_703.f9 CaMgO_full_703.f9
mv CaMgO_full_tmp_703.gui CaMgO_full_703.gui
mv CaMgO_full_tmp_704.f9 CaMgO_full_704.f9
mv CaMgO_full_tmp_704.gui CaMgO_full_704.gui
mv CaMgO_full_tmp_705.f9 CaMgO_full_705.f9
mv CaMgO_full_tmp_705.gui CaMgO_full_705.gui
mv CaMgO_full_tmp_706.f9 CaMgO_full_706.f9
mv CaMgO_full_tmp_706.gui CaMgO_full_706.gui
mv CaMgO_full_tmp_707.f9 CaMgO_full_707.f9
mv CaMgO_full_tmp_707.gui CaMgO_full_707.gui
mv CaMgO_full_tmp_708.f9 CaMgO_full_708.f9
mv CaMgO_full_tmp_708.gui CaMgO_full_708.gui
mv CaMgO_full_tmp_709.f9 CaMgO_full_709.f9
mv CaMgO_full_tmp_709.gui CaMgO_full_709.gui
mv CaMgO_full_tmp_710.f9 CaMgO_full_710.f9
mv CaMgO_full_tmp_710.gui CaMgO_full_710.gui
mv CaMgO_full_tmp_711.f9 CaMgO_full_711.f9
mv CaMgO_full_tmp_711.gui CaMgO_full_711.gui
mv CaMgO_full_tmp_712.f9 CaMgO_full_712.f9
mv CaMgO_full_tmp_712.gui CaMgO_full_712.gui
mv CaMgO_full_tmp_713.f9 CaMgO_full_713.f9
mv CaMgO_full_tmp_713.gui CaMgO_full_713.gui
mv CaMgO_full_tmp_714.f9 CaMgO_full_714.f9
mv CaMgO_full_tmp_714.gui CaMgO_full_714.gui
mv CaMgO_full_tmp_715.f9 CaMgO_full_715.f9
mv CaMgO_full_tmp_715.gui CaMgO_full_715.gui
mv CaMgO_full_tmp_716.f9 CaMgO_full_716.f9
mv CaMgO_full_tmp_716.gui CaMgO_full_716.gui
mv CaMgO_full_tmp_717.f9 CaMgO_full_717.f9
mv CaMgO_full_tmp_717.gui CaMgO_full_717.gui
mv CaMgO_full_tmp_718.f9 CaMgO_full_718.f9
mv CaMgO_full_tmp_718.gui CaMgO_full_718.gui
mv CaMgO_full_tmp_719.f9 CaMgO_full_719.f9
mv CaMgO_full_tmp_719.gui CaMgO_full_719.gui
mv CaMgO_full_tmp_720.f9 CaMgO_full_720.f9
mv CaMgO_full_tmp_720.gui CaMgO_full_720.gui
mv CaMgO_full_tmp_721.f9 CaMgO_full_721.f9
mv CaMgO_full_tmp_721.gui CaMgO_full_721.gui
mv CaMgO_full_tmp_722.f9 CaMgO_full_722.f9
mv CaMgO_full_tmp_722.gui CaMgO_full_722.gui
mv CaMgO_full_tmp_723.f9 CaMgO_full_723.f9
mv CaMgO_full_tmp_723.gui CaMgO_full_723.gui
mv CaMgO_full_tmp_724.f9 CaMgO_full_724.f9
mv CaMgO_full_tmp_724.gui CaMgO_full_724.gui
mv CaMgO_full_tmp_725.f9 CaMgO_full_725.f9
mv CaMgO_full_tmp_725.gui CaMgO_full_725.gui
mv CaMgO_full_tmp_726.f9 CaMgO_full_726.f9
mv CaMgO_full_tmp_726.gui CaMgO_full_726.gui
mv CaMgO_full_tmp_727.f9 CaMgO_full_727.f9
mv CaMgO_full_tmp_727.gui CaMgO_full_727.gui
mv CaMgO_full_tmp_728.f9 CaMgO_full_728.f9
mv CaMgO_full_tmp_728.gui CaMgO_full_728.gui
mv CaMgO_full_tmp_729.f9 CaMgO_full_729.f9
mv CaMgO_full_tmp_729.gui CaMgO_full_729.gui
mv CaMgO_full_tmp_730.f9 CaMgO_full_730.f9
mv CaMgO_full_tmp_730.gui CaMgO_full_730.gui
mv CaMgO_full_tmp_731.f9 CaMgO_full_731.f9
mv CaMgO_full_tmp_731.gui CaMgO_full_731.gui
mv CaMgO_full_tmp_732.f9 CaMgO_full_732.f9
mv CaMgO_full_tmp_732.gui CaMgO_full_732.gui
mv CaMgO_full_tmp_733.f9 CaMgO_full_733.f9
mv CaMgO_full_tmp_733.gui CaMgO_full_733.gui
mv CaMgO_full_tmp_734.f9 CaMgO_full_734.f9
mv CaMgO_full_tmp_734.gui CaMgO_full_734.gui
mv CaMgO_full_tmp_735.f9 CaMgO_full_735.f9
mv CaMgO_full_tmp_735.gui CaMgO_full_735.gui
mv CaMgO_full_tmp_736.f9 CaMgO_full_736.f9
mv CaMgO_full_tmp_736.gui CaMgO_full_736.gui
mv CaMgO_full_tmp_737.f9 CaMgO_full_737.f9
mv CaMgO_full_tmp_737.gui CaMgO_full_737.gui
mv CaMgO_full_tmp_738.f9 CaMgO_full_738.f9
mv CaMgO_full_tmp_738.gui CaMgO_full_738.gui
mv CaMgO_full_tmp_739.f9 CaMgO_full_739.f9
mv CaMgO_full_tmp_739.gui CaMgO_full_739.gui
mv CaMgO_full_tmp_740.f9 CaMgO_full_740.f9
mv CaMgO_full_tmp_740.gui CaMgO_full_740.gui
mv CaMgO_full_tmp_741.f9 CaMgO_full_741.f9
mv CaMgO_full_tmp_741.gui CaMgO_full_741.gui
mv CaMgO_full_tmp_742.f9 CaMgO_full_742.f9
mv CaMgO_full_tmp_742.gui CaMgO_full_742.gui
mv CaMgO_full_tmp_743.f9 CaMgO_full_743.f9
mv CaMgO_full_tmp_743.gui CaMgO_full_743.gui
mv CaMgO_full_tmp_744.f9 CaMgO_full_744.f9
mv CaMgO_full_tmp_744.gui CaMgO_full_744.gui
mv CaMgO_full_tmp_745.f9 CaMgO_full_745.f9
mv CaMgO_full_tmp_745.gui CaMgO_full_745.gui
mv CaMgO_full_tmp_746.f9 CaMgO_full_746.f9
mv CaMgO_full_tmp_746.gui CaMgO_full_746.gui
mv CaMgO_full_tmp_747.f9 CaMgO_full_747.f9
mv CaMgO_full_tmp_747.gui CaMgO_full_747.gui
mv CaMgO_full_tmp_748.f9 CaMgO_full_748.f9
mv CaMgO_full_tmp_748.gui CaMgO_full_748.gui
mv CaMgO_full_tmp_749.f9 CaMgO_full_749.f9
mv CaMgO_full_tmp_749.gui CaMgO_full_749.gui
mv CaMgO_full_tmp_750.f9 CaMgO_full_750.f9
mv CaMgO_full_tmp_750.gui CaMgO_full_750.gui
mv CaMgO_full_tmp_751.f9 CaMgO_full_751.f9
mv CaMgO_full_tmp_751.gui CaMgO_full_751.gui
mv CaMgO_full_tmp_752.f9 CaMgO_full_752.f9
mv CaMgO_full_tmp_752.gui CaMgO_full_752.gui
mv CaMgO_full_tmp_753.f9 CaMgO_full_753.f9
mv CaMgO_full_tmp_753.gui CaMgO_full_753.gui
mv CaMgO_full_tmp_754.f9 CaMgO_full_754.f9
mv CaMgO_full_tmp_754.gui CaMgO_full_754.gui
mv CaMgO_full_tmp_755.f9 CaMgO_full_755.f9
mv CaMgO_full_tmp_755.gui CaMgO_full_755.gui
mv CaMgO_full_tmp_756.f9 CaMgO_full_756.f9
mv CaMgO_full_tmp_756.gui CaMgO_full_756.gui
mv CaMgO_full_tmp_757.f9 CaMgO_full_757.f9
mv CaMgO_full_tmp_757.gui CaMgO_full_757.gui
mv CaMgO_full_tmp_758.f9 CaMgO_full_758.f9
mv CaMgO_full_tmp_758.gui CaMgO_full_758.gui
mv CaMgO_full_tmp_759.f9 CaMgO_full_759.f9
mv CaMgO_full_tmp_759.gui CaMgO_full_759.gui
mv CaMgO_full_tmp_760.f9 CaMgO_full_760.f9
mv CaMgO_full_tmp_760.gui CaMgO_full_760.gui
mv CaMgO_full_tmp_761.f9 CaMgO_full_761.f9
mv CaMgO_full_tmp_761.gui CaMgO_full_761.gui
mv CaMgO_full_tmp_762.f9 CaMgO_full_762.f9
mv CaMgO_full_tmp_762.gui CaMgO_full_762.gui
mv CaMgO_full_tmp_763.f9 CaMgO_full_763.f9
mv CaMgO_full_tmp_763.gui CaMgO_full_763.gui
mv CaMgO_full_tmp_764.f9 CaMgO_full_764.f9
mv CaMgO_full_tmp_764.gui CaMgO_full_764.gui
mv CaMgO_full_tmp_765.f9 CaMgO_full_765.f9
mv CaMgO_full_tmp_765.gui CaMgO_full_765.gui
mv CaMgO_full_tmp_766.f9 CaMgO_full_766.f9
mv CaMgO_full_tmp_766.gui CaMgO_full_766.gui
mv CaMgO_full_tmp_767.f9 CaMgO_full_767.f9
mv CaMgO_full_tmp_767.gui CaMgO_full_767.gui
mv CaMgO_full_tmp_768.f9 CaMgO_full_768.f9
mv CaMgO_full_tmp_768.gui CaMgO_full_768.gui
mv CaMgO_full_tmp_769.f9 CaMgO_full_769.f9
mv CaMgO_full_tmp_769.gui CaMgO_full_769.gui
mv CaMgO_full_tmp_770.f9 CaMgO_full_770.f9
mv CaMgO_full_tmp_770.gui CaMgO_full_770.gui
mv CaMgO_full_tmp_771.f9 CaMgO_full_771.f9
mv CaMgO_full_tmp_771.gui CaMgO_full_771.gui
mv CaMgO_full_tmp_772.f9 CaMgO_full_772.f9
mv CaMgO_full_tmp_772.gui CaMgO_full_772.gui
mv CaMgO_full_tmp_773.f9 CaMgO_full_773.f9
mv CaMgO_full_tmp_773.gui CaMgO_full_773.gui
mv CaMgO_full_tmp_774.f9 CaMgO_full_774.f9
mv CaMgO_full_tmp_774.gui CaMgO_full_774.gui
mv CaMgO_full_tmp_775.f9 CaMgO_full_775.f9
mv CaMgO_full_tmp_775.gui CaMgO_full_775.gui
mv CaMgO_full_tmp_776.f9 CaMgO_full_776.f9
mv CaMgO_full_tmp_776.gui CaMgO_full_776.gui
mv CaMgO_full_tmp_777.f9 CaMgO_full_777.f9
mv CaMgO_full_tmp_777.gui CaMgO_full_777.gui
mv CaMgO_full_tmp_778.f9 CaMgO_full_778.f9
mv CaMgO_full_tmp_778.gui CaMgO_full_778.gui
mv CaMgO_full_tmp_779.f9 CaMgO_full_779.f9
mv CaMgO_full_tmp_779.gui CaMgO_full_779.gui
mv CaMgO_full_tmp_780.f9 CaMgO_full_780.f9
mv CaMgO_full_tmp_780.gui CaMgO_full_780.gui
mv CaMgO_full_tmp_781.f9 CaMgO_full_781.f9
mv CaMgO_full_tmp_781.gui CaMgO_full_781.gui
mv CaMgO_full_tmp_782.f9 CaMgO_full_782.f9
mv CaMgO_full_tmp_782.gui CaMgO_full_782.gui
mv CaMgO_full_tmp_783.f9 CaMgO_full_783.f9
mv CaMgO_full_tmp_783.gui CaMgO_full_783.gui
mv CaMgO_full_tmp_784.f9 CaMgO_full_784.f9
mv CaMgO_full_tmp_784.gui CaMgO_full_784.gui
mv CaMgO_full_tmp_785.f9 CaMgO_full_785.f9
mv CaMgO_full_tmp_785.gui CaMgO_full_785.gui
mv CaMgO_full_tmp_786.f9 CaMgO_full_786.f9
mv CaMgO_full_tmp_786.gui CaMgO_full_786.gui
mv CaMgO_full_tmp_787.f9 CaMgO_full_787.f9
mv CaMgO_full_tmp_787.gui CaMgO_full_787.gui
mv CaMgO_full_tmp_788.f9 CaMgO_full_788.f9
mv CaMgO_full_tmp_788.gui CaMgO_full_788.gui
mv CaMgO_full_tmp_789.f9 CaMgO_full_789.f9
mv CaMgO_full_tmp_789.gui CaMgO_full_789.gui
mv CaMgO_full_tmp_790.f9 CaMgO_full_790.f9
mv CaMgO_full_tmp_790.gui CaMgO_full_790.gui
mv CaMgO_full_tmp_791.f9 CaMgO_full_791.f9
mv CaMgO_full_tmp_791.gui CaMgO_full_791.gui
mv CaMgO_full_tmp_792.f9 CaMgO_full_792.f9
mv CaMgO_full_tmp_792.gui CaMgO_full_792.gui
mv CaMgO_full_tmp_793.f9 CaMgO_full_793.f9
mv CaMgO_full_tmp_793.gui CaMgO_full_793.gui
mv CaMgO_full_tmp_794.f9 CaMgO_full_794.f9
mv CaMgO_full_tmp_794.gui CaMgO_full_794.gui
mv CaMgO_full_tmp_795.f9 CaMgO_full_795.f9
mv CaMgO_full_tmp_795.gui CaMgO_full_795.gui
mv CaMgO_full_tmp_796.f9 CaMgO_full_796.f9
mv CaMgO_full_tmp_796.gui CaMgO_full_796.gui
mv CaMgO_full_tmp_797.f9 CaMgO_full_797.f9
mv CaMgO_full_tmp_797.gui CaMgO_full_797.gui
mv CaMgO_full_tmp_798.f9 CaMgO_full_798.f9
mv CaMgO_full_tmp_798.gui CaMgO_full_798.gui
mv CaMgO_full_tmp_799.f9 CaMgO_full_799.f9
mv CaMgO_full_tmp_799.gui CaMgO_full_799.gui
