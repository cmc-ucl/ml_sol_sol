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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_801 CaMgO_coord_tmp_801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_802 CaMgO_coord_tmp_802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_803 CaMgO_coord_tmp_803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_804 CaMgO_coord_tmp_804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_805 CaMgO_coord_tmp_805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_806 CaMgO_coord_tmp_806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_807 CaMgO_coord_tmp_807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_808 CaMgO_coord_tmp_808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_809 CaMgO_coord_tmp_809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_810 CaMgO_coord_tmp_810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_811 CaMgO_coord_tmp_811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_812 CaMgO_coord_tmp_812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_813 CaMgO_coord_tmp_813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_814 CaMgO_coord_tmp_814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_815 CaMgO_coord_tmp_815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_816 CaMgO_coord_tmp_816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_817 CaMgO_coord_tmp_817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_818 CaMgO_coord_tmp_818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_819 CaMgO_coord_tmp_819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_820 CaMgO_coord_tmp_820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_821 CaMgO_coord_tmp_821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_822 CaMgO_coord_tmp_822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_823 CaMgO_coord_tmp_823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_824 CaMgO_coord_tmp_824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_825 CaMgO_coord_tmp_825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_826 CaMgO_coord_tmp_826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_827 CaMgO_coord_tmp_827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_828 CaMgO_coord_tmp_828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_829 CaMgO_coord_tmp_829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_830 CaMgO_coord_tmp_830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_831 CaMgO_coord_tmp_831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_832 CaMgO_coord_tmp_832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_833 CaMgO_coord_tmp_833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_834 CaMgO_coord_tmp_834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_835 CaMgO_coord_tmp_835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_836 CaMgO_coord_tmp_836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_837 CaMgO_coord_tmp_837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_838 CaMgO_coord_tmp_838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_839 CaMgO_coord_tmp_839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_840 CaMgO_coord_tmp_840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_841 CaMgO_coord_tmp_841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_842 CaMgO_coord_tmp_842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_843 CaMgO_coord_tmp_843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_844 CaMgO_coord_tmp_844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_845 CaMgO_coord_tmp_845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_846 CaMgO_coord_tmp_846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_847 CaMgO_coord_tmp_847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_848 CaMgO_coord_tmp_848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_849 CaMgO_coord_tmp_849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_850 CaMgO_coord_tmp_850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_851 CaMgO_coord_tmp_851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_852 CaMgO_coord_tmp_852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_853 CaMgO_coord_tmp_853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_854 CaMgO_coord_tmp_854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_855 CaMgO_coord_tmp_855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_856 CaMgO_coord_tmp_856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_857 CaMgO_coord_tmp_857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_858 CaMgO_coord_tmp_858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_859 CaMgO_coord_tmp_859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_860 CaMgO_coord_tmp_860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_861 CaMgO_coord_tmp_861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_862 CaMgO_coord_tmp_862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_863 CaMgO_coord_tmp_863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_864 CaMgO_coord_tmp_864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_865 CaMgO_coord_tmp_865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_866 CaMgO_coord_tmp_866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_867 CaMgO_coord_tmp_867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_868 CaMgO_coord_tmp_868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_869 CaMgO_coord_tmp_869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_870 CaMgO_coord_tmp_870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_871 CaMgO_coord_tmp_871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_872 CaMgO_coord_tmp_872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_873 CaMgO_coord_tmp_873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_874 CaMgO_coord_tmp_874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_875 CaMgO_coord_tmp_875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_876 CaMgO_coord_tmp_876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_877 CaMgO_coord_tmp_877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_878 CaMgO_coord_tmp_878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_879 CaMgO_coord_tmp_879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_880 CaMgO_coord_tmp_880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_881 CaMgO_coord_tmp_881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_882 CaMgO_coord_tmp_882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_883 CaMgO_coord_tmp_883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_884 CaMgO_coord_tmp_884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_885 CaMgO_coord_tmp_885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_886 CaMgO_coord_tmp_886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_887 CaMgO_coord_tmp_887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_888 CaMgO_coord_tmp_888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_889 CaMgO_coord_tmp_889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_890 CaMgO_coord_tmp_890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_891 CaMgO_coord_tmp_891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_892 CaMgO_coord_tmp_892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_893 CaMgO_coord_tmp_893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_894 CaMgO_coord_tmp_894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_895 CaMgO_coord_tmp_895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_896 CaMgO_coord_tmp_896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_897 CaMgO_coord_tmp_897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_898 CaMgO_coord_tmp_898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_899 CaMgO_coord_tmp_899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_899


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_tmp_900 CaMgO_coord_tmp_900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_tmp_900


