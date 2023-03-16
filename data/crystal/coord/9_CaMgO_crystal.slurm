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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_801 CaMgO_coord_801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_802 CaMgO_coord_802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_803 CaMgO_coord_803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_804 CaMgO_coord_804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_805 CaMgO_coord_805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_806 CaMgO_coord_806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_807 CaMgO_coord_807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_808 CaMgO_coord_808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_809 CaMgO_coord_809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_810 CaMgO_coord_810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_811 CaMgO_coord_811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_812 CaMgO_coord_812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_813 CaMgO_coord_813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_814 CaMgO_coord_814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_815 CaMgO_coord_815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_816 CaMgO_coord_816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_817 CaMgO_coord_817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_818 CaMgO_coord_818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_819 CaMgO_coord_819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_820 CaMgO_coord_820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_821 CaMgO_coord_821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_822 CaMgO_coord_822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_823 CaMgO_coord_823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_824 CaMgO_coord_824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_825 CaMgO_coord_825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_826 CaMgO_coord_826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_827 CaMgO_coord_827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_828 CaMgO_coord_828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_829 CaMgO_coord_829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_830 CaMgO_coord_830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_831 CaMgO_coord_831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_832 CaMgO_coord_832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_833 CaMgO_coord_833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_834 CaMgO_coord_834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_835 CaMgO_coord_835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_836 CaMgO_coord_836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_837 CaMgO_coord_837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_838 CaMgO_coord_838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_839 CaMgO_coord_839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_840 CaMgO_coord_840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_841 CaMgO_coord_841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_842 CaMgO_coord_842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_843 CaMgO_coord_843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_844 CaMgO_coord_844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_845 CaMgO_coord_845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_846 CaMgO_coord_846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_847 CaMgO_coord_847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_848 CaMgO_coord_848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_849 CaMgO_coord_849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_850 CaMgO_coord_850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_851 CaMgO_coord_851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_852 CaMgO_coord_852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_853 CaMgO_coord_853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_854 CaMgO_coord_854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_855 CaMgO_coord_855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_856 CaMgO_coord_856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_857 CaMgO_coord_857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_858 CaMgO_coord_858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_859 CaMgO_coord_859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_860 CaMgO_coord_860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_861 CaMgO_coord_861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_862 CaMgO_coord_862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_863 CaMgO_coord_863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_864 CaMgO_coord_864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_865 CaMgO_coord_865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_866 CaMgO_coord_866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_867 CaMgO_coord_867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_868 CaMgO_coord_868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_869 CaMgO_coord_869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_870 CaMgO_coord_870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_871 CaMgO_coord_871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_872 CaMgO_coord_872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_873 CaMgO_coord_873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_874 CaMgO_coord_874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_875 CaMgO_coord_875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_876 CaMgO_coord_876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_877 CaMgO_coord_877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_878 CaMgO_coord_878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_879 CaMgO_coord_879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_880 CaMgO_coord_880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_881 CaMgO_coord_881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_882 CaMgO_coord_882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_883 CaMgO_coord_883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_884 CaMgO_coord_884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_885 CaMgO_coord_885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_886 CaMgO_coord_886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_887 CaMgO_coord_887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_888 CaMgO_coord_888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_889 CaMgO_coord_889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_890 CaMgO_coord_890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_891 CaMgO_coord_891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_892 CaMgO_coord_892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_893 CaMgO_coord_893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_894 CaMgO_coord_894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_895 CaMgO_coord_895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_896 CaMgO_coord_896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_897 CaMgO_coord_897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_898 CaMgO_coord_898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_899 CaMgO_coord_899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_899


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_900 CaMgO_coord_900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_900


