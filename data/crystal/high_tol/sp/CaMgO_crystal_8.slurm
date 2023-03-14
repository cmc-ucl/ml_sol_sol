#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-gc-smw    
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_700


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_701
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_701


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_702
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_702


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_703
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_703


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_704
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_704


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_705
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_705


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_706
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_706


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_707
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_707


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_708
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_708


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_709
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_709


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_710
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_710


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_711
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_711


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_712
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_712


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_713
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_713


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_714
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_714


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_715
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_715


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_716
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_716


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_717
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_717


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_718
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_718


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_719
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_719


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_720
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_720


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_721
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_721


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_722
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_722


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_723
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_723


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_724
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_724


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_725
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_725


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_726
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_726


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_727
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_727


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_728
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_728


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_729
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_729


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_730
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_730


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_731
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_731


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_732
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_732


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_733
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_733


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_734
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_734


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_735
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_735


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_736
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_736


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_737
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_737


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_738
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_738


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_739
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_739


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_740
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_740


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_741
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_741


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_742
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_742


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_743
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_743


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_744
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_744


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_745
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_745


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_746
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_746


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_747
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_747


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_748
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_748


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_749
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_749


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_750
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_750


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_751
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_751


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_752
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_752


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_753
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_753


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_754
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_754


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_755
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_755


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_756
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_756


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_757
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_757


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_758
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_758


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_759
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_759


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_760
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_760


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_761
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_761


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_762
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_762


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_763
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_763


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_764
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_764


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_765
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_765


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_766
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_766


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_767
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_767


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_768
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_768


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_769
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_769


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_770
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_770


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_771
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_771


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_772
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_772


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_773
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_773


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_774
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_774


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_775
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_775


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_776
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_776


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_777
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_777


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_778
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_778


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_779
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_779


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_780
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_780


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_781
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_781


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_782
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_782


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_783
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_783


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_784
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_784


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_785
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_785


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_786
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_786


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_787
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_787


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_788
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_788


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_789
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_789


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_790
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_790


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_791
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_791


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_792
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_792


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_793
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_793


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_794
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_794


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_795
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_795


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_796
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_796


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_797
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_797


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_798
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_798


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_799
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_799


