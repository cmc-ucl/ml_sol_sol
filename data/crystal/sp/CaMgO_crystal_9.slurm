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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_801
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_801


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_802
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_802


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_803
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_803


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_804
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_804


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_805
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_805


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_806
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_806


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_807
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_807


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_808
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_808


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_809
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_809


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_810
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_810


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_811
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_811


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_812
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_812


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_813
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_813


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_814
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_814


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_815
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_815


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_816
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_816


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_817
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_817


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_818
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_818


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_819
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_819


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_820
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_820


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_821
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_821


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_822
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_822


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_823
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_823


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_824
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_824


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_825
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_825


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_826
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_826


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_827
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_827


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_828
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_828


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_829
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_829


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_830
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_830


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_831
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_831


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_832
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_832


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_833
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_833


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_834
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_834


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_835
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_835


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_836
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_836


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_837
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_837


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_838
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_838


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_839
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_839


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_840
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_840


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_841
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_841


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_842
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_842


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_843
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_843


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_844
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_844


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_845
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_845


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_846
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_846


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_847
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_847


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_848
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_848


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_849
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_849


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_850
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_850


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_851
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_851


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_852
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_852


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_853
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_853


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_854
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_854


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_855
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_855


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_856
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_856


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_857
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_857


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_858
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_858


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_859
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_859


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_860
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_860


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_861
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_861


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_862
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_862


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_863
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_863


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_864
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_864


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_865
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_865


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_866
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_866


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_867
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_867


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_868
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_868


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_869
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_869


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_870
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_870


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_871
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_871


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_872
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_872


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_873
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_873


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_874
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_874


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_875
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_875


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_876
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_876


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_877
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_877


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_878
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_878


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_879
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_879


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_880
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_880


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_881
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_881


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_882
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_882


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_883
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_883


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_884
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_884


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_885
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_885


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_886
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_886


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_887
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_887


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_888
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_888


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_889
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_889


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_890
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_890


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_891
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_891


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_892
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_892


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_893
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_893


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_894
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_894


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_895
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_895


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_896
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_896


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_897
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_897


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_898
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_898


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_899
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_899


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_900
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_900


