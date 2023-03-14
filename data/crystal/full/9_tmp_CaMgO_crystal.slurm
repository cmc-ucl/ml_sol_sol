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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_800 CaMgO_full_tmp_800 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_801 CaMgO_full_tmp_801 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_802 CaMgO_full_tmp_802 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_803 CaMgO_full_tmp_803 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_804 CaMgO_full_tmp_804 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_805 CaMgO_full_tmp_805 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_806 CaMgO_full_tmp_806 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_807 CaMgO_full_tmp_807 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_808 CaMgO_full_tmp_808 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_809 CaMgO_full_tmp_809 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_810 CaMgO_full_tmp_810 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_811 CaMgO_full_tmp_811 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_812 CaMgO_full_tmp_812 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_813 CaMgO_full_tmp_813 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_814 CaMgO_full_tmp_814 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_815 CaMgO_full_tmp_815 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_816 CaMgO_full_tmp_816 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_817 CaMgO_full_tmp_817 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_818 CaMgO_full_tmp_818 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_819 CaMgO_full_tmp_819 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_820 CaMgO_full_tmp_820 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_821 CaMgO_full_tmp_821 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_822 CaMgO_full_tmp_822 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_823 CaMgO_full_tmp_823 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_824 CaMgO_full_tmp_824 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_825 CaMgO_full_tmp_825 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_826 CaMgO_full_tmp_826 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_827 CaMgO_full_tmp_827 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_828 CaMgO_full_tmp_828 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_829 CaMgO_full_tmp_829 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_830 CaMgO_full_tmp_830 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_831 CaMgO_full_tmp_831 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_832 CaMgO_full_tmp_832 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_833 CaMgO_full_tmp_833 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_834 CaMgO_full_tmp_834 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_835 CaMgO_full_tmp_835 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_836 CaMgO_full_tmp_836 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_837 CaMgO_full_tmp_837 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_838 CaMgO_full_tmp_838 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_839 CaMgO_full_tmp_839 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_840 CaMgO_full_tmp_840 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_841 CaMgO_full_tmp_841 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_842 CaMgO_full_tmp_842 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_843 CaMgO_full_tmp_843 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_844 CaMgO_full_tmp_844 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_845 CaMgO_full_tmp_845 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_846 CaMgO_full_tmp_846 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_847 CaMgO_full_tmp_847 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_848 CaMgO_full_tmp_848 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_849 CaMgO_full_tmp_849 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_850 CaMgO_full_tmp_850 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_851 CaMgO_full_tmp_851 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_852 CaMgO_full_tmp_852 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_853 CaMgO_full_tmp_853 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_854 CaMgO_full_tmp_854 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_855 CaMgO_full_tmp_855 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_856 CaMgO_full_tmp_856 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_857 CaMgO_full_tmp_857 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_858 CaMgO_full_tmp_858 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_859 CaMgO_full_tmp_859 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_860 CaMgO_full_tmp_860 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_861 CaMgO_full_tmp_861 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_862 CaMgO_full_tmp_862 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_863 CaMgO_full_tmp_863 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_864 CaMgO_full_tmp_864 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_865 CaMgO_full_tmp_865 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_866 CaMgO_full_tmp_866 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_867 CaMgO_full_tmp_867 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_868 CaMgO_full_tmp_868 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_869 CaMgO_full_tmp_869 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_870 CaMgO_full_tmp_870 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_871 CaMgO_full_tmp_871 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_872 CaMgO_full_tmp_872 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_873 CaMgO_full_tmp_873 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_874 CaMgO_full_tmp_874 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_875 CaMgO_full_tmp_875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_876 CaMgO_full_tmp_876 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_877 CaMgO_full_tmp_877 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_878 CaMgO_full_tmp_878 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_879 CaMgO_full_tmp_879 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_880 CaMgO_full_tmp_880 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_881 CaMgO_full_tmp_881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_882 CaMgO_full_tmp_882 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_883 CaMgO_full_tmp_883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_884 CaMgO_full_tmp_884 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_885 CaMgO_full_tmp_885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_886 CaMgO_full_tmp_886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_887 CaMgO_full_tmp_887 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_888 CaMgO_full_tmp_888 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_889 CaMgO_full_tmp_889 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_890 CaMgO_full_tmp_890 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_891 CaMgO_full_tmp_891 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_892 CaMgO_full_tmp_892 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_893 CaMgO_full_tmp_893 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_894 CaMgO_full_tmp_894 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_895 CaMgO_full_tmp_895 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_896 CaMgO_full_tmp_896 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_897 CaMgO_full_tmp_897 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_898 CaMgO_full_tmp_898 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_899 CaMgO_full_tmp_899 &

wait
mv CaMgO_full_tmp_800.f9 CaMgO_full_800.f9
mv CaMgO_full_tmp_800.gui CaMgO_full_800.gui
mv CaMgO_full_tmp_801.f9 CaMgO_full_801.f9
mv CaMgO_full_tmp_801.gui CaMgO_full_801.gui
mv CaMgO_full_tmp_802.f9 CaMgO_full_802.f9
mv CaMgO_full_tmp_802.gui CaMgO_full_802.gui
mv CaMgO_full_tmp_803.f9 CaMgO_full_803.f9
mv CaMgO_full_tmp_803.gui CaMgO_full_803.gui
mv CaMgO_full_tmp_804.f9 CaMgO_full_804.f9
mv CaMgO_full_tmp_804.gui CaMgO_full_804.gui
mv CaMgO_full_tmp_805.f9 CaMgO_full_805.f9
mv CaMgO_full_tmp_805.gui CaMgO_full_805.gui
mv CaMgO_full_tmp_806.f9 CaMgO_full_806.f9
mv CaMgO_full_tmp_806.gui CaMgO_full_806.gui
mv CaMgO_full_tmp_807.f9 CaMgO_full_807.f9
mv CaMgO_full_tmp_807.gui CaMgO_full_807.gui
mv CaMgO_full_tmp_808.f9 CaMgO_full_808.f9
mv CaMgO_full_tmp_808.gui CaMgO_full_808.gui
mv CaMgO_full_tmp_809.f9 CaMgO_full_809.f9
mv CaMgO_full_tmp_809.gui CaMgO_full_809.gui
mv CaMgO_full_tmp_810.f9 CaMgO_full_810.f9
mv CaMgO_full_tmp_810.gui CaMgO_full_810.gui
mv CaMgO_full_tmp_811.f9 CaMgO_full_811.f9
mv CaMgO_full_tmp_811.gui CaMgO_full_811.gui
mv CaMgO_full_tmp_812.f9 CaMgO_full_812.f9
mv CaMgO_full_tmp_812.gui CaMgO_full_812.gui
mv CaMgO_full_tmp_813.f9 CaMgO_full_813.f9
mv CaMgO_full_tmp_813.gui CaMgO_full_813.gui
mv CaMgO_full_tmp_814.f9 CaMgO_full_814.f9
mv CaMgO_full_tmp_814.gui CaMgO_full_814.gui
mv CaMgO_full_tmp_815.f9 CaMgO_full_815.f9
mv CaMgO_full_tmp_815.gui CaMgO_full_815.gui
mv CaMgO_full_tmp_816.f9 CaMgO_full_816.f9
mv CaMgO_full_tmp_816.gui CaMgO_full_816.gui
mv CaMgO_full_tmp_817.f9 CaMgO_full_817.f9
mv CaMgO_full_tmp_817.gui CaMgO_full_817.gui
mv CaMgO_full_tmp_818.f9 CaMgO_full_818.f9
mv CaMgO_full_tmp_818.gui CaMgO_full_818.gui
mv CaMgO_full_tmp_819.f9 CaMgO_full_819.f9
mv CaMgO_full_tmp_819.gui CaMgO_full_819.gui
mv CaMgO_full_tmp_820.f9 CaMgO_full_820.f9
mv CaMgO_full_tmp_820.gui CaMgO_full_820.gui
mv CaMgO_full_tmp_821.f9 CaMgO_full_821.f9
mv CaMgO_full_tmp_821.gui CaMgO_full_821.gui
mv CaMgO_full_tmp_822.f9 CaMgO_full_822.f9
mv CaMgO_full_tmp_822.gui CaMgO_full_822.gui
mv CaMgO_full_tmp_823.f9 CaMgO_full_823.f9
mv CaMgO_full_tmp_823.gui CaMgO_full_823.gui
mv CaMgO_full_tmp_824.f9 CaMgO_full_824.f9
mv CaMgO_full_tmp_824.gui CaMgO_full_824.gui
mv CaMgO_full_tmp_825.f9 CaMgO_full_825.f9
mv CaMgO_full_tmp_825.gui CaMgO_full_825.gui
mv CaMgO_full_tmp_826.f9 CaMgO_full_826.f9
mv CaMgO_full_tmp_826.gui CaMgO_full_826.gui
mv CaMgO_full_tmp_827.f9 CaMgO_full_827.f9
mv CaMgO_full_tmp_827.gui CaMgO_full_827.gui
mv CaMgO_full_tmp_828.f9 CaMgO_full_828.f9
mv CaMgO_full_tmp_828.gui CaMgO_full_828.gui
mv CaMgO_full_tmp_829.f9 CaMgO_full_829.f9
mv CaMgO_full_tmp_829.gui CaMgO_full_829.gui
mv CaMgO_full_tmp_830.f9 CaMgO_full_830.f9
mv CaMgO_full_tmp_830.gui CaMgO_full_830.gui
mv CaMgO_full_tmp_831.f9 CaMgO_full_831.f9
mv CaMgO_full_tmp_831.gui CaMgO_full_831.gui
mv CaMgO_full_tmp_832.f9 CaMgO_full_832.f9
mv CaMgO_full_tmp_832.gui CaMgO_full_832.gui
mv CaMgO_full_tmp_833.f9 CaMgO_full_833.f9
mv CaMgO_full_tmp_833.gui CaMgO_full_833.gui
mv CaMgO_full_tmp_834.f9 CaMgO_full_834.f9
mv CaMgO_full_tmp_834.gui CaMgO_full_834.gui
mv CaMgO_full_tmp_835.f9 CaMgO_full_835.f9
mv CaMgO_full_tmp_835.gui CaMgO_full_835.gui
mv CaMgO_full_tmp_836.f9 CaMgO_full_836.f9
mv CaMgO_full_tmp_836.gui CaMgO_full_836.gui
mv CaMgO_full_tmp_837.f9 CaMgO_full_837.f9
mv CaMgO_full_tmp_837.gui CaMgO_full_837.gui
mv CaMgO_full_tmp_838.f9 CaMgO_full_838.f9
mv CaMgO_full_tmp_838.gui CaMgO_full_838.gui
mv CaMgO_full_tmp_839.f9 CaMgO_full_839.f9
mv CaMgO_full_tmp_839.gui CaMgO_full_839.gui
mv CaMgO_full_tmp_840.f9 CaMgO_full_840.f9
mv CaMgO_full_tmp_840.gui CaMgO_full_840.gui
mv CaMgO_full_tmp_841.f9 CaMgO_full_841.f9
mv CaMgO_full_tmp_841.gui CaMgO_full_841.gui
mv CaMgO_full_tmp_842.f9 CaMgO_full_842.f9
mv CaMgO_full_tmp_842.gui CaMgO_full_842.gui
mv CaMgO_full_tmp_843.f9 CaMgO_full_843.f9
mv CaMgO_full_tmp_843.gui CaMgO_full_843.gui
mv CaMgO_full_tmp_844.f9 CaMgO_full_844.f9
mv CaMgO_full_tmp_844.gui CaMgO_full_844.gui
mv CaMgO_full_tmp_845.f9 CaMgO_full_845.f9
mv CaMgO_full_tmp_845.gui CaMgO_full_845.gui
mv CaMgO_full_tmp_846.f9 CaMgO_full_846.f9
mv CaMgO_full_tmp_846.gui CaMgO_full_846.gui
mv CaMgO_full_tmp_847.f9 CaMgO_full_847.f9
mv CaMgO_full_tmp_847.gui CaMgO_full_847.gui
mv CaMgO_full_tmp_848.f9 CaMgO_full_848.f9
mv CaMgO_full_tmp_848.gui CaMgO_full_848.gui
mv CaMgO_full_tmp_849.f9 CaMgO_full_849.f9
mv CaMgO_full_tmp_849.gui CaMgO_full_849.gui
mv CaMgO_full_tmp_850.f9 CaMgO_full_850.f9
mv CaMgO_full_tmp_850.gui CaMgO_full_850.gui
mv CaMgO_full_tmp_851.f9 CaMgO_full_851.f9
mv CaMgO_full_tmp_851.gui CaMgO_full_851.gui
mv CaMgO_full_tmp_852.f9 CaMgO_full_852.f9
mv CaMgO_full_tmp_852.gui CaMgO_full_852.gui
mv CaMgO_full_tmp_853.f9 CaMgO_full_853.f9
mv CaMgO_full_tmp_853.gui CaMgO_full_853.gui
mv CaMgO_full_tmp_854.f9 CaMgO_full_854.f9
mv CaMgO_full_tmp_854.gui CaMgO_full_854.gui
mv CaMgO_full_tmp_855.f9 CaMgO_full_855.f9
mv CaMgO_full_tmp_855.gui CaMgO_full_855.gui
mv CaMgO_full_tmp_856.f9 CaMgO_full_856.f9
mv CaMgO_full_tmp_856.gui CaMgO_full_856.gui
mv CaMgO_full_tmp_857.f9 CaMgO_full_857.f9
mv CaMgO_full_tmp_857.gui CaMgO_full_857.gui
mv CaMgO_full_tmp_858.f9 CaMgO_full_858.f9
mv CaMgO_full_tmp_858.gui CaMgO_full_858.gui
mv CaMgO_full_tmp_859.f9 CaMgO_full_859.f9
mv CaMgO_full_tmp_859.gui CaMgO_full_859.gui
mv CaMgO_full_tmp_860.f9 CaMgO_full_860.f9
mv CaMgO_full_tmp_860.gui CaMgO_full_860.gui
mv CaMgO_full_tmp_861.f9 CaMgO_full_861.f9
mv CaMgO_full_tmp_861.gui CaMgO_full_861.gui
mv CaMgO_full_tmp_862.f9 CaMgO_full_862.f9
mv CaMgO_full_tmp_862.gui CaMgO_full_862.gui
mv CaMgO_full_tmp_863.f9 CaMgO_full_863.f9
mv CaMgO_full_tmp_863.gui CaMgO_full_863.gui
mv CaMgO_full_tmp_864.f9 CaMgO_full_864.f9
mv CaMgO_full_tmp_864.gui CaMgO_full_864.gui
mv CaMgO_full_tmp_865.f9 CaMgO_full_865.f9
mv CaMgO_full_tmp_865.gui CaMgO_full_865.gui
mv CaMgO_full_tmp_866.f9 CaMgO_full_866.f9
mv CaMgO_full_tmp_866.gui CaMgO_full_866.gui
mv CaMgO_full_tmp_867.f9 CaMgO_full_867.f9
mv CaMgO_full_tmp_867.gui CaMgO_full_867.gui
mv CaMgO_full_tmp_868.f9 CaMgO_full_868.f9
mv CaMgO_full_tmp_868.gui CaMgO_full_868.gui
mv CaMgO_full_tmp_869.f9 CaMgO_full_869.f9
mv CaMgO_full_tmp_869.gui CaMgO_full_869.gui
mv CaMgO_full_tmp_870.f9 CaMgO_full_870.f9
mv CaMgO_full_tmp_870.gui CaMgO_full_870.gui
mv CaMgO_full_tmp_871.f9 CaMgO_full_871.f9
mv CaMgO_full_tmp_871.gui CaMgO_full_871.gui
mv CaMgO_full_tmp_872.f9 CaMgO_full_872.f9
mv CaMgO_full_tmp_872.gui CaMgO_full_872.gui
mv CaMgO_full_tmp_873.f9 CaMgO_full_873.f9
mv CaMgO_full_tmp_873.gui CaMgO_full_873.gui
mv CaMgO_full_tmp_874.f9 CaMgO_full_874.f9
mv CaMgO_full_tmp_874.gui CaMgO_full_874.gui
mv CaMgO_full_tmp_875.f9 CaMgO_full_875.f9
mv CaMgO_full_tmp_875.gui CaMgO_full_875.gui
mv CaMgO_full_tmp_876.f9 CaMgO_full_876.f9
mv CaMgO_full_tmp_876.gui CaMgO_full_876.gui
mv CaMgO_full_tmp_877.f9 CaMgO_full_877.f9
mv CaMgO_full_tmp_877.gui CaMgO_full_877.gui
mv CaMgO_full_tmp_878.f9 CaMgO_full_878.f9
mv CaMgO_full_tmp_878.gui CaMgO_full_878.gui
mv CaMgO_full_tmp_879.f9 CaMgO_full_879.f9
mv CaMgO_full_tmp_879.gui CaMgO_full_879.gui
mv CaMgO_full_tmp_880.f9 CaMgO_full_880.f9
mv CaMgO_full_tmp_880.gui CaMgO_full_880.gui
mv CaMgO_full_tmp_881.f9 CaMgO_full_881.f9
mv CaMgO_full_tmp_881.gui CaMgO_full_881.gui
mv CaMgO_full_tmp_882.f9 CaMgO_full_882.f9
mv CaMgO_full_tmp_882.gui CaMgO_full_882.gui
mv CaMgO_full_tmp_883.f9 CaMgO_full_883.f9
mv CaMgO_full_tmp_883.gui CaMgO_full_883.gui
mv CaMgO_full_tmp_884.f9 CaMgO_full_884.f9
mv CaMgO_full_tmp_884.gui CaMgO_full_884.gui
mv CaMgO_full_tmp_885.f9 CaMgO_full_885.f9
mv CaMgO_full_tmp_885.gui CaMgO_full_885.gui
mv CaMgO_full_tmp_886.f9 CaMgO_full_886.f9
mv CaMgO_full_tmp_886.gui CaMgO_full_886.gui
mv CaMgO_full_tmp_887.f9 CaMgO_full_887.f9
mv CaMgO_full_tmp_887.gui CaMgO_full_887.gui
mv CaMgO_full_tmp_888.f9 CaMgO_full_888.f9
mv CaMgO_full_tmp_888.gui CaMgO_full_888.gui
mv CaMgO_full_tmp_889.f9 CaMgO_full_889.f9
mv CaMgO_full_tmp_889.gui CaMgO_full_889.gui
mv CaMgO_full_tmp_890.f9 CaMgO_full_890.f9
mv CaMgO_full_tmp_890.gui CaMgO_full_890.gui
mv CaMgO_full_tmp_891.f9 CaMgO_full_891.f9
mv CaMgO_full_tmp_891.gui CaMgO_full_891.gui
mv CaMgO_full_tmp_892.f9 CaMgO_full_892.f9
mv CaMgO_full_tmp_892.gui CaMgO_full_892.gui
mv CaMgO_full_tmp_893.f9 CaMgO_full_893.f9
mv CaMgO_full_tmp_893.gui CaMgO_full_893.gui
mv CaMgO_full_tmp_894.f9 CaMgO_full_894.f9
mv CaMgO_full_tmp_894.gui CaMgO_full_894.gui
mv CaMgO_full_tmp_895.f9 CaMgO_full_895.f9
mv CaMgO_full_tmp_895.gui CaMgO_full_895.gui
mv CaMgO_full_tmp_896.f9 CaMgO_full_896.f9
mv CaMgO_full_tmp_896.gui CaMgO_full_896.gui
mv CaMgO_full_tmp_897.f9 CaMgO_full_897.f9
mv CaMgO_full_tmp_897.gui CaMgO_full_897.gui
mv CaMgO_full_tmp_898.f9 CaMgO_full_898.f9
mv CaMgO_full_tmp_898.gui CaMgO_full_898.gui
mv CaMgO_full_tmp_899.f9 CaMgO_full_899.f9
mv CaMgO_full_tmp_899.gui CaMgO_full_899.gui
