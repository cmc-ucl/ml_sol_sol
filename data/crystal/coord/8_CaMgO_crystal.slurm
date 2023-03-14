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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_701 CaMgO_coord_701
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_701


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_702 CaMgO_coord_702
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_702


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_703 CaMgO_coord_703
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_703


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_704 CaMgO_coord_704
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_704


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_705 CaMgO_coord_705
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_705


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_706 CaMgO_coord_706
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_706


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_707 CaMgO_coord_707
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_707


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_708 CaMgO_coord_708
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_708


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_709 CaMgO_coord_709
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_709


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_710 CaMgO_coord_710
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_710


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_711 CaMgO_coord_711
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_711


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_712 CaMgO_coord_712
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_712


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_713 CaMgO_coord_713
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_713


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_714 CaMgO_coord_714
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_714


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_715 CaMgO_coord_715
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_715


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_716 CaMgO_coord_716
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_716


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_717 CaMgO_coord_717
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_717


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_718 CaMgO_coord_718
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_718


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_719 CaMgO_coord_719
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_719


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_720 CaMgO_coord_720
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_720


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_721 CaMgO_coord_721
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_721


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_722 CaMgO_coord_722
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_722


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_723 CaMgO_coord_723
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_723


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_724 CaMgO_coord_724
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_724


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_725 CaMgO_coord_725
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_725


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_726 CaMgO_coord_726
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_726


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_727 CaMgO_coord_727
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_727


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_728 CaMgO_coord_728
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_728


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_729 CaMgO_coord_729
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_729


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_730 CaMgO_coord_730
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_730


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_731 CaMgO_coord_731
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_731


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_732 CaMgO_coord_732
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_732


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_733 CaMgO_coord_733
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_733


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_734 CaMgO_coord_734
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_734


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_735 CaMgO_coord_735
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_735


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_736 CaMgO_coord_736
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_736


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_737 CaMgO_coord_737
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_737


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_738 CaMgO_coord_738
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_738


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_739 CaMgO_coord_739
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_739


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_740 CaMgO_coord_740
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_740


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_741 CaMgO_coord_741
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_741


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_742 CaMgO_coord_742
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_742


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_743 CaMgO_coord_743
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_743


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_744 CaMgO_coord_744
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_744


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_745 CaMgO_coord_745
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_745


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_746 CaMgO_coord_746
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_746


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_747 CaMgO_coord_747
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_747


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_748 CaMgO_coord_748
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_748


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_749 CaMgO_coord_749
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_749


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_750 CaMgO_coord_750
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_750


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_751 CaMgO_coord_751
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_751


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_752 CaMgO_coord_752
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_752


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_753 CaMgO_coord_753
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_753


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_754 CaMgO_coord_754
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_754


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_755 CaMgO_coord_755
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_755


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_756 CaMgO_coord_756
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_756


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_757 CaMgO_coord_757
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_757


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_758 CaMgO_coord_758
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_758


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_759 CaMgO_coord_759
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_759


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_760 CaMgO_coord_760
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_760


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_761 CaMgO_coord_761
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_761


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_762 CaMgO_coord_762
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_762


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_763 CaMgO_coord_763
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_763


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_764 CaMgO_coord_764
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_764


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_765 CaMgO_coord_765
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_765


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_766 CaMgO_coord_766
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_766


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_767 CaMgO_coord_767
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_767


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_768 CaMgO_coord_768
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_768


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_769 CaMgO_coord_769
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_769


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_770 CaMgO_coord_770
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_770


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_771 CaMgO_coord_771
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_771


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_772 CaMgO_coord_772
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_772


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_773 CaMgO_coord_773
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_773


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_774 CaMgO_coord_774
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_774


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_775 CaMgO_coord_775
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_775


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_776 CaMgO_coord_776
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_776


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_777 CaMgO_coord_777
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_777


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_778 CaMgO_coord_778
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_778


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_779 CaMgO_coord_779
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_779


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_780 CaMgO_coord_780
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_780


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_781 CaMgO_coord_781
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_781


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_782 CaMgO_coord_782
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_782


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_783 CaMgO_coord_783
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_783


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_784 CaMgO_coord_784
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_784


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_785 CaMgO_coord_785
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_785


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_786 CaMgO_coord_786
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_786


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_787 CaMgO_coord_787
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_787


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_788 CaMgO_coord_788
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_788


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_789 CaMgO_coord_789
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_789


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_790 CaMgO_coord_790
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_790


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_791 CaMgO_coord_791
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_791


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_792 CaMgO_coord_792
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_792


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_793 CaMgO_coord_793
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_793


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_794 CaMgO_coord_794
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_794


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_795 CaMgO_coord_795
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_795


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_796 CaMgO_coord_796
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_796


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_797 CaMgO_coord_797
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_797


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_798 CaMgO_coord_798
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_798


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_799 CaMgO_coord_799
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_799


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_800 CaMgO_coord_800
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_800


