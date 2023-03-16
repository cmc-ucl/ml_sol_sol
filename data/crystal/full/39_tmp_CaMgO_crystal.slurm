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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3800 CaMgO_full_tmp_3800 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3801 CaMgO_full_tmp_3801 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3802 CaMgO_full_tmp_3802 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3803 CaMgO_full_tmp_3803 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3804 CaMgO_full_tmp_3804 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3805 CaMgO_full_tmp_3805 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3806 CaMgO_full_tmp_3806 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3807 CaMgO_full_tmp_3807 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3808 CaMgO_full_tmp_3808 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3809 CaMgO_full_tmp_3809 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3810 CaMgO_full_tmp_3810 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3811 CaMgO_full_tmp_3811 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3812 CaMgO_full_tmp_3812 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3813 CaMgO_full_tmp_3813 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3814 CaMgO_full_tmp_3814 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3815 CaMgO_full_tmp_3815 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3816 CaMgO_full_tmp_3816 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3817 CaMgO_full_tmp_3817 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3818 CaMgO_full_tmp_3818 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3819 CaMgO_full_tmp_3819 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3820 CaMgO_full_tmp_3820 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3821 CaMgO_full_tmp_3821 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3822 CaMgO_full_tmp_3822 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3823 CaMgO_full_tmp_3823 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3824 CaMgO_full_tmp_3824 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3825 CaMgO_full_tmp_3825 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3826 CaMgO_full_tmp_3826 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3827 CaMgO_full_tmp_3827 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3828 CaMgO_full_tmp_3828 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3829 CaMgO_full_tmp_3829 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3830 CaMgO_full_tmp_3830 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3831 CaMgO_full_tmp_3831 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3832 CaMgO_full_tmp_3832 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3833 CaMgO_full_tmp_3833 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3834 CaMgO_full_tmp_3834 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3835 CaMgO_full_tmp_3835 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3836 CaMgO_full_tmp_3836 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3837 CaMgO_full_tmp_3837 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3838 CaMgO_full_tmp_3838 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3839 CaMgO_full_tmp_3839 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3840 CaMgO_full_tmp_3840 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3841 CaMgO_full_tmp_3841 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3842 CaMgO_full_tmp_3842 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3843 CaMgO_full_tmp_3843 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3844 CaMgO_full_tmp_3844 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3845 CaMgO_full_tmp_3845 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3846 CaMgO_full_tmp_3846 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3847 CaMgO_full_tmp_3847 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3848 CaMgO_full_tmp_3848 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3849 CaMgO_full_tmp_3849 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3850 CaMgO_full_tmp_3850 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3851 CaMgO_full_tmp_3851 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3852 CaMgO_full_tmp_3852 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3853 CaMgO_full_tmp_3853 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3854 CaMgO_full_tmp_3854 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3855 CaMgO_full_tmp_3855 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3856 CaMgO_full_tmp_3856 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3857 CaMgO_full_tmp_3857 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3858 CaMgO_full_tmp_3858 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3859 CaMgO_full_tmp_3859 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3860 CaMgO_full_tmp_3860 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3861 CaMgO_full_tmp_3861 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3862 CaMgO_full_tmp_3862 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3863 CaMgO_full_tmp_3863 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3864 CaMgO_full_tmp_3864 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3865 CaMgO_full_tmp_3865 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3866 CaMgO_full_tmp_3866 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3867 CaMgO_full_tmp_3867 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3868 CaMgO_full_tmp_3868 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3869 CaMgO_full_tmp_3869 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3870 CaMgO_full_tmp_3870 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3871 CaMgO_full_tmp_3871 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3872 CaMgO_full_tmp_3872 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3873 CaMgO_full_tmp_3873 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3874 CaMgO_full_tmp_3874 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3875 CaMgO_full_tmp_3875 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3876 CaMgO_full_tmp_3876 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3877 CaMgO_full_tmp_3877 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3878 CaMgO_full_tmp_3878 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3879 CaMgO_full_tmp_3879 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3880 CaMgO_full_tmp_3880 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3881 CaMgO_full_tmp_3881 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3882 CaMgO_full_tmp_3882 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3883 CaMgO_full_tmp_3883 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3884 CaMgO_full_tmp_3884 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3885 CaMgO_full_tmp_3885 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3886 CaMgO_full_tmp_3886 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3887 CaMgO_full_tmp_3887 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3888 CaMgO_full_tmp_3888 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3889 CaMgO_full_tmp_3889 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3890 CaMgO_full_tmp_3890 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3891 CaMgO_full_tmp_3891 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3892 CaMgO_full_tmp_3892 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3893 CaMgO_full_tmp_3893 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3894 CaMgO_full_tmp_3894 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3895 CaMgO_full_tmp_3895 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3896 CaMgO_full_tmp_3896 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3897 CaMgO_full_tmp_3897 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3898 CaMgO_full_tmp_3898 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3899 CaMgO_full_tmp_3899 &

wait
mv CaMgO_full_tmp_3800.f9 CaMgO_full_3800.f9
mv CaMgO_full_tmp_3800.gui CaMgO_full_3800.gui
mv CaMgO_full_tmp_3801.f9 CaMgO_full_3801.f9
mv CaMgO_full_tmp_3801.gui CaMgO_full_3801.gui
mv CaMgO_full_tmp_3802.f9 CaMgO_full_3802.f9
mv CaMgO_full_tmp_3802.gui CaMgO_full_3802.gui
mv CaMgO_full_tmp_3803.f9 CaMgO_full_3803.f9
mv CaMgO_full_tmp_3803.gui CaMgO_full_3803.gui
mv CaMgO_full_tmp_3804.f9 CaMgO_full_3804.f9
mv CaMgO_full_tmp_3804.gui CaMgO_full_3804.gui
mv CaMgO_full_tmp_3805.f9 CaMgO_full_3805.f9
mv CaMgO_full_tmp_3805.gui CaMgO_full_3805.gui
mv CaMgO_full_tmp_3806.f9 CaMgO_full_3806.f9
mv CaMgO_full_tmp_3806.gui CaMgO_full_3806.gui
mv CaMgO_full_tmp_3807.f9 CaMgO_full_3807.f9
mv CaMgO_full_tmp_3807.gui CaMgO_full_3807.gui
mv CaMgO_full_tmp_3808.f9 CaMgO_full_3808.f9
mv CaMgO_full_tmp_3808.gui CaMgO_full_3808.gui
mv CaMgO_full_tmp_3809.f9 CaMgO_full_3809.f9
mv CaMgO_full_tmp_3809.gui CaMgO_full_3809.gui
mv CaMgO_full_tmp_3810.f9 CaMgO_full_3810.f9
mv CaMgO_full_tmp_3810.gui CaMgO_full_3810.gui
mv CaMgO_full_tmp_3811.f9 CaMgO_full_3811.f9
mv CaMgO_full_tmp_3811.gui CaMgO_full_3811.gui
mv CaMgO_full_tmp_3812.f9 CaMgO_full_3812.f9
mv CaMgO_full_tmp_3812.gui CaMgO_full_3812.gui
mv CaMgO_full_tmp_3813.f9 CaMgO_full_3813.f9
mv CaMgO_full_tmp_3813.gui CaMgO_full_3813.gui
mv CaMgO_full_tmp_3814.f9 CaMgO_full_3814.f9
mv CaMgO_full_tmp_3814.gui CaMgO_full_3814.gui
mv CaMgO_full_tmp_3815.f9 CaMgO_full_3815.f9
mv CaMgO_full_tmp_3815.gui CaMgO_full_3815.gui
mv CaMgO_full_tmp_3816.f9 CaMgO_full_3816.f9
mv CaMgO_full_tmp_3816.gui CaMgO_full_3816.gui
mv CaMgO_full_tmp_3817.f9 CaMgO_full_3817.f9
mv CaMgO_full_tmp_3817.gui CaMgO_full_3817.gui
mv CaMgO_full_tmp_3818.f9 CaMgO_full_3818.f9
mv CaMgO_full_tmp_3818.gui CaMgO_full_3818.gui
mv CaMgO_full_tmp_3819.f9 CaMgO_full_3819.f9
mv CaMgO_full_tmp_3819.gui CaMgO_full_3819.gui
mv CaMgO_full_tmp_3820.f9 CaMgO_full_3820.f9
mv CaMgO_full_tmp_3820.gui CaMgO_full_3820.gui
mv CaMgO_full_tmp_3821.f9 CaMgO_full_3821.f9
mv CaMgO_full_tmp_3821.gui CaMgO_full_3821.gui
mv CaMgO_full_tmp_3822.f9 CaMgO_full_3822.f9
mv CaMgO_full_tmp_3822.gui CaMgO_full_3822.gui
mv CaMgO_full_tmp_3823.f9 CaMgO_full_3823.f9
mv CaMgO_full_tmp_3823.gui CaMgO_full_3823.gui
mv CaMgO_full_tmp_3824.f9 CaMgO_full_3824.f9
mv CaMgO_full_tmp_3824.gui CaMgO_full_3824.gui
mv CaMgO_full_tmp_3825.f9 CaMgO_full_3825.f9
mv CaMgO_full_tmp_3825.gui CaMgO_full_3825.gui
mv CaMgO_full_tmp_3826.f9 CaMgO_full_3826.f9
mv CaMgO_full_tmp_3826.gui CaMgO_full_3826.gui
mv CaMgO_full_tmp_3827.f9 CaMgO_full_3827.f9
mv CaMgO_full_tmp_3827.gui CaMgO_full_3827.gui
mv CaMgO_full_tmp_3828.f9 CaMgO_full_3828.f9
mv CaMgO_full_tmp_3828.gui CaMgO_full_3828.gui
mv CaMgO_full_tmp_3829.f9 CaMgO_full_3829.f9
mv CaMgO_full_tmp_3829.gui CaMgO_full_3829.gui
mv CaMgO_full_tmp_3830.f9 CaMgO_full_3830.f9
mv CaMgO_full_tmp_3830.gui CaMgO_full_3830.gui
mv CaMgO_full_tmp_3831.f9 CaMgO_full_3831.f9
mv CaMgO_full_tmp_3831.gui CaMgO_full_3831.gui
mv CaMgO_full_tmp_3832.f9 CaMgO_full_3832.f9
mv CaMgO_full_tmp_3832.gui CaMgO_full_3832.gui
mv CaMgO_full_tmp_3833.f9 CaMgO_full_3833.f9
mv CaMgO_full_tmp_3833.gui CaMgO_full_3833.gui
mv CaMgO_full_tmp_3834.f9 CaMgO_full_3834.f9
mv CaMgO_full_tmp_3834.gui CaMgO_full_3834.gui
mv CaMgO_full_tmp_3835.f9 CaMgO_full_3835.f9
mv CaMgO_full_tmp_3835.gui CaMgO_full_3835.gui
mv CaMgO_full_tmp_3836.f9 CaMgO_full_3836.f9
mv CaMgO_full_tmp_3836.gui CaMgO_full_3836.gui
mv CaMgO_full_tmp_3837.f9 CaMgO_full_3837.f9
mv CaMgO_full_tmp_3837.gui CaMgO_full_3837.gui
mv CaMgO_full_tmp_3838.f9 CaMgO_full_3838.f9
mv CaMgO_full_tmp_3838.gui CaMgO_full_3838.gui
mv CaMgO_full_tmp_3839.f9 CaMgO_full_3839.f9
mv CaMgO_full_tmp_3839.gui CaMgO_full_3839.gui
mv CaMgO_full_tmp_3840.f9 CaMgO_full_3840.f9
mv CaMgO_full_tmp_3840.gui CaMgO_full_3840.gui
mv CaMgO_full_tmp_3841.f9 CaMgO_full_3841.f9
mv CaMgO_full_tmp_3841.gui CaMgO_full_3841.gui
mv CaMgO_full_tmp_3842.f9 CaMgO_full_3842.f9
mv CaMgO_full_tmp_3842.gui CaMgO_full_3842.gui
mv CaMgO_full_tmp_3843.f9 CaMgO_full_3843.f9
mv CaMgO_full_tmp_3843.gui CaMgO_full_3843.gui
mv CaMgO_full_tmp_3844.f9 CaMgO_full_3844.f9
mv CaMgO_full_tmp_3844.gui CaMgO_full_3844.gui
mv CaMgO_full_tmp_3845.f9 CaMgO_full_3845.f9
mv CaMgO_full_tmp_3845.gui CaMgO_full_3845.gui
mv CaMgO_full_tmp_3846.f9 CaMgO_full_3846.f9
mv CaMgO_full_tmp_3846.gui CaMgO_full_3846.gui
mv CaMgO_full_tmp_3847.f9 CaMgO_full_3847.f9
mv CaMgO_full_tmp_3847.gui CaMgO_full_3847.gui
mv CaMgO_full_tmp_3848.f9 CaMgO_full_3848.f9
mv CaMgO_full_tmp_3848.gui CaMgO_full_3848.gui
mv CaMgO_full_tmp_3849.f9 CaMgO_full_3849.f9
mv CaMgO_full_tmp_3849.gui CaMgO_full_3849.gui
mv CaMgO_full_tmp_3850.f9 CaMgO_full_3850.f9
mv CaMgO_full_tmp_3850.gui CaMgO_full_3850.gui
mv CaMgO_full_tmp_3851.f9 CaMgO_full_3851.f9
mv CaMgO_full_tmp_3851.gui CaMgO_full_3851.gui
mv CaMgO_full_tmp_3852.f9 CaMgO_full_3852.f9
mv CaMgO_full_tmp_3852.gui CaMgO_full_3852.gui
mv CaMgO_full_tmp_3853.f9 CaMgO_full_3853.f9
mv CaMgO_full_tmp_3853.gui CaMgO_full_3853.gui
mv CaMgO_full_tmp_3854.f9 CaMgO_full_3854.f9
mv CaMgO_full_tmp_3854.gui CaMgO_full_3854.gui
mv CaMgO_full_tmp_3855.f9 CaMgO_full_3855.f9
mv CaMgO_full_tmp_3855.gui CaMgO_full_3855.gui
mv CaMgO_full_tmp_3856.f9 CaMgO_full_3856.f9
mv CaMgO_full_tmp_3856.gui CaMgO_full_3856.gui
mv CaMgO_full_tmp_3857.f9 CaMgO_full_3857.f9
mv CaMgO_full_tmp_3857.gui CaMgO_full_3857.gui
mv CaMgO_full_tmp_3858.f9 CaMgO_full_3858.f9
mv CaMgO_full_tmp_3858.gui CaMgO_full_3858.gui
mv CaMgO_full_tmp_3859.f9 CaMgO_full_3859.f9
mv CaMgO_full_tmp_3859.gui CaMgO_full_3859.gui
mv CaMgO_full_tmp_3860.f9 CaMgO_full_3860.f9
mv CaMgO_full_tmp_3860.gui CaMgO_full_3860.gui
mv CaMgO_full_tmp_3861.f9 CaMgO_full_3861.f9
mv CaMgO_full_tmp_3861.gui CaMgO_full_3861.gui
mv CaMgO_full_tmp_3862.f9 CaMgO_full_3862.f9
mv CaMgO_full_tmp_3862.gui CaMgO_full_3862.gui
mv CaMgO_full_tmp_3863.f9 CaMgO_full_3863.f9
mv CaMgO_full_tmp_3863.gui CaMgO_full_3863.gui
mv CaMgO_full_tmp_3864.f9 CaMgO_full_3864.f9
mv CaMgO_full_tmp_3864.gui CaMgO_full_3864.gui
mv CaMgO_full_tmp_3865.f9 CaMgO_full_3865.f9
mv CaMgO_full_tmp_3865.gui CaMgO_full_3865.gui
mv CaMgO_full_tmp_3866.f9 CaMgO_full_3866.f9
mv CaMgO_full_tmp_3866.gui CaMgO_full_3866.gui
mv CaMgO_full_tmp_3867.f9 CaMgO_full_3867.f9
mv CaMgO_full_tmp_3867.gui CaMgO_full_3867.gui
mv CaMgO_full_tmp_3868.f9 CaMgO_full_3868.f9
mv CaMgO_full_tmp_3868.gui CaMgO_full_3868.gui
mv CaMgO_full_tmp_3869.f9 CaMgO_full_3869.f9
mv CaMgO_full_tmp_3869.gui CaMgO_full_3869.gui
mv CaMgO_full_tmp_3870.f9 CaMgO_full_3870.f9
mv CaMgO_full_tmp_3870.gui CaMgO_full_3870.gui
mv CaMgO_full_tmp_3871.f9 CaMgO_full_3871.f9
mv CaMgO_full_tmp_3871.gui CaMgO_full_3871.gui
mv CaMgO_full_tmp_3872.f9 CaMgO_full_3872.f9
mv CaMgO_full_tmp_3872.gui CaMgO_full_3872.gui
mv CaMgO_full_tmp_3873.f9 CaMgO_full_3873.f9
mv CaMgO_full_tmp_3873.gui CaMgO_full_3873.gui
mv CaMgO_full_tmp_3874.f9 CaMgO_full_3874.f9
mv CaMgO_full_tmp_3874.gui CaMgO_full_3874.gui
mv CaMgO_full_tmp_3875.f9 CaMgO_full_3875.f9
mv CaMgO_full_tmp_3875.gui CaMgO_full_3875.gui
mv CaMgO_full_tmp_3876.f9 CaMgO_full_3876.f9
mv CaMgO_full_tmp_3876.gui CaMgO_full_3876.gui
mv CaMgO_full_tmp_3877.f9 CaMgO_full_3877.f9
mv CaMgO_full_tmp_3877.gui CaMgO_full_3877.gui
mv CaMgO_full_tmp_3878.f9 CaMgO_full_3878.f9
mv CaMgO_full_tmp_3878.gui CaMgO_full_3878.gui
mv CaMgO_full_tmp_3879.f9 CaMgO_full_3879.f9
mv CaMgO_full_tmp_3879.gui CaMgO_full_3879.gui
mv CaMgO_full_tmp_3880.f9 CaMgO_full_3880.f9
mv CaMgO_full_tmp_3880.gui CaMgO_full_3880.gui
mv CaMgO_full_tmp_3881.f9 CaMgO_full_3881.f9
mv CaMgO_full_tmp_3881.gui CaMgO_full_3881.gui
mv CaMgO_full_tmp_3882.f9 CaMgO_full_3882.f9
mv CaMgO_full_tmp_3882.gui CaMgO_full_3882.gui
mv CaMgO_full_tmp_3883.f9 CaMgO_full_3883.f9
mv CaMgO_full_tmp_3883.gui CaMgO_full_3883.gui
mv CaMgO_full_tmp_3884.f9 CaMgO_full_3884.f9
mv CaMgO_full_tmp_3884.gui CaMgO_full_3884.gui
mv CaMgO_full_tmp_3885.f9 CaMgO_full_3885.f9
mv CaMgO_full_tmp_3885.gui CaMgO_full_3885.gui
mv CaMgO_full_tmp_3886.f9 CaMgO_full_3886.f9
mv CaMgO_full_tmp_3886.gui CaMgO_full_3886.gui
mv CaMgO_full_tmp_3887.f9 CaMgO_full_3887.f9
mv CaMgO_full_tmp_3887.gui CaMgO_full_3887.gui
mv CaMgO_full_tmp_3888.f9 CaMgO_full_3888.f9
mv CaMgO_full_tmp_3888.gui CaMgO_full_3888.gui
mv CaMgO_full_tmp_3889.f9 CaMgO_full_3889.f9
mv CaMgO_full_tmp_3889.gui CaMgO_full_3889.gui
mv CaMgO_full_tmp_3890.f9 CaMgO_full_3890.f9
mv CaMgO_full_tmp_3890.gui CaMgO_full_3890.gui
mv CaMgO_full_tmp_3891.f9 CaMgO_full_3891.f9
mv CaMgO_full_tmp_3891.gui CaMgO_full_3891.gui
mv CaMgO_full_tmp_3892.f9 CaMgO_full_3892.f9
mv CaMgO_full_tmp_3892.gui CaMgO_full_3892.gui
mv CaMgO_full_tmp_3893.f9 CaMgO_full_3893.f9
mv CaMgO_full_tmp_3893.gui CaMgO_full_3893.gui
mv CaMgO_full_tmp_3894.f9 CaMgO_full_3894.f9
mv CaMgO_full_tmp_3894.gui CaMgO_full_3894.gui
mv CaMgO_full_tmp_3895.f9 CaMgO_full_3895.f9
mv CaMgO_full_tmp_3895.gui CaMgO_full_3895.gui
mv CaMgO_full_tmp_3896.f9 CaMgO_full_3896.f9
mv CaMgO_full_tmp_3896.gui CaMgO_full_3896.gui
mv CaMgO_full_tmp_3897.f9 CaMgO_full_3897.f9
mv CaMgO_full_tmp_3897.gui CaMgO_full_3897.gui
mv CaMgO_full_tmp_3898.f9 CaMgO_full_3898.f9
mv CaMgO_full_tmp_3898.gui CaMgO_full_3898.gui
mv CaMgO_full_tmp_3899.f9 CaMgO_full_3899.f9
mv CaMgO_full_tmp_3899.gui CaMgO_full_3899.gui
