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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3900 CaMgO_full_tmp_3900 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3901 CaMgO_full_tmp_3901 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3902 CaMgO_full_tmp_3902 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3903 CaMgO_full_tmp_3903 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3904 CaMgO_full_tmp_3904 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3905 CaMgO_full_tmp_3905 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3906 CaMgO_full_tmp_3906 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3907 CaMgO_full_tmp_3907 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3908 CaMgO_full_tmp_3908 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3909 CaMgO_full_tmp_3909 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3910 CaMgO_full_tmp_3910 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3911 CaMgO_full_tmp_3911 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3912 CaMgO_full_tmp_3912 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3913 CaMgO_full_tmp_3913 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3914 CaMgO_full_tmp_3914 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3915 CaMgO_full_tmp_3915 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3916 CaMgO_full_tmp_3916 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3917 CaMgO_full_tmp_3917 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3918 CaMgO_full_tmp_3918 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3919 CaMgO_full_tmp_3919 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3920 CaMgO_full_tmp_3920 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3921 CaMgO_full_tmp_3921 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3922 CaMgO_full_tmp_3922 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3923 CaMgO_full_tmp_3923 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3924 CaMgO_full_tmp_3924 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3925 CaMgO_full_tmp_3925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3926 CaMgO_full_tmp_3926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3927 CaMgO_full_tmp_3927 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3928 CaMgO_full_tmp_3928 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3929 CaMgO_full_tmp_3929 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3930 CaMgO_full_tmp_3930 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3931 CaMgO_full_tmp_3931 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3932 CaMgO_full_tmp_3932 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3933 CaMgO_full_tmp_3933 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3934 CaMgO_full_tmp_3934 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3935 CaMgO_full_tmp_3935 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3936 CaMgO_full_tmp_3936 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3937 CaMgO_full_tmp_3937 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3938 CaMgO_full_tmp_3938 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3939 CaMgO_full_tmp_3939 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3940 CaMgO_full_tmp_3940 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3941 CaMgO_full_tmp_3941 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3942 CaMgO_full_tmp_3942 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3943 CaMgO_full_tmp_3943 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3944 CaMgO_full_tmp_3944 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3945 CaMgO_full_tmp_3945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3946 CaMgO_full_tmp_3946 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3947 CaMgO_full_tmp_3947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3948 CaMgO_full_tmp_3948 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3949 CaMgO_full_tmp_3949 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3950 CaMgO_full_tmp_3950 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3951 CaMgO_full_tmp_3951 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3952 CaMgO_full_tmp_3952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3953 CaMgO_full_tmp_3953 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3954 CaMgO_full_tmp_3954 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3955 CaMgO_full_tmp_3955 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3956 CaMgO_full_tmp_3956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3957 CaMgO_full_tmp_3957 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3958 CaMgO_full_tmp_3958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3959 CaMgO_full_tmp_3959 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3960 CaMgO_full_tmp_3960 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3961 CaMgO_full_tmp_3961 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3962 CaMgO_full_tmp_3962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3963 CaMgO_full_tmp_3963 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3964 CaMgO_full_tmp_3964 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3965 CaMgO_full_tmp_3965 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3966 CaMgO_full_tmp_3966 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3967 CaMgO_full_tmp_3967 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3968 CaMgO_full_tmp_3968 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3969 CaMgO_full_tmp_3969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3970 CaMgO_full_tmp_3970 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3971 CaMgO_full_tmp_3971 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3972 CaMgO_full_tmp_3972 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3973 CaMgO_full_tmp_3973 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3974 CaMgO_full_tmp_3974 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3975 CaMgO_full_tmp_3975 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3976 CaMgO_full_tmp_3976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3977 CaMgO_full_tmp_3977 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3978 CaMgO_full_tmp_3978 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3979 CaMgO_full_tmp_3979 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3980 CaMgO_full_tmp_3980 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3981 CaMgO_full_tmp_3981 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3982 CaMgO_full_tmp_3982 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3983 CaMgO_full_tmp_3983 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3984 CaMgO_full_tmp_3984 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3985 CaMgO_full_tmp_3985 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3986 CaMgO_full_tmp_3986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3987 CaMgO_full_tmp_3987 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3988 CaMgO_full_tmp_3988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3989 CaMgO_full_tmp_3989 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3990 CaMgO_full_tmp_3990 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3991 CaMgO_full_tmp_3991 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3992 CaMgO_full_tmp_3992 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3993 CaMgO_full_tmp_3993 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3994 CaMgO_full_tmp_3994 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3995 CaMgO_full_tmp_3995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3996 CaMgO_full_tmp_3996 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3997 CaMgO_full_tmp_3997 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3998 CaMgO_full_tmp_3998 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3999 CaMgO_full_tmp_3999 &

wait
mv CaMgO_full_tmp_3900.f9 CaMgO_full_3900.f9
mv CaMgO_full_tmp_3900.gui CaMgO_full_3900.gui
mv CaMgO_full_tmp_3901.f9 CaMgO_full_3901.f9
mv CaMgO_full_tmp_3901.gui CaMgO_full_3901.gui
mv CaMgO_full_tmp_3902.f9 CaMgO_full_3902.f9
mv CaMgO_full_tmp_3902.gui CaMgO_full_3902.gui
mv CaMgO_full_tmp_3903.f9 CaMgO_full_3903.f9
mv CaMgO_full_tmp_3903.gui CaMgO_full_3903.gui
mv CaMgO_full_tmp_3904.f9 CaMgO_full_3904.f9
mv CaMgO_full_tmp_3904.gui CaMgO_full_3904.gui
mv CaMgO_full_tmp_3905.f9 CaMgO_full_3905.f9
mv CaMgO_full_tmp_3905.gui CaMgO_full_3905.gui
mv CaMgO_full_tmp_3906.f9 CaMgO_full_3906.f9
mv CaMgO_full_tmp_3906.gui CaMgO_full_3906.gui
mv CaMgO_full_tmp_3907.f9 CaMgO_full_3907.f9
mv CaMgO_full_tmp_3907.gui CaMgO_full_3907.gui
mv CaMgO_full_tmp_3908.f9 CaMgO_full_3908.f9
mv CaMgO_full_tmp_3908.gui CaMgO_full_3908.gui
mv CaMgO_full_tmp_3909.f9 CaMgO_full_3909.f9
mv CaMgO_full_tmp_3909.gui CaMgO_full_3909.gui
mv CaMgO_full_tmp_3910.f9 CaMgO_full_3910.f9
mv CaMgO_full_tmp_3910.gui CaMgO_full_3910.gui
mv CaMgO_full_tmp_3911.f9 CaMgO_full_3911.f9
mv CaMgO_full_tmp_3911.gui CaMgO_full_3911.gui
mv CaMgO_full_tmp_3912.f9 CaMgO_full_3912.f9
mv CaMgO_full_tmp_3912.gui CaMgO_full_3912.gui
mv CaMgO_full_tmp_3913.f9 CaMgO_full_3913.f9
mv CaMgO_full_tmp_3913.gui CaMgO_full_3913.gui
mv CaMgO_full_tmp_3914.f9 CaMgO_full_3914.f9
mv CaMgO_full_tmp_3914.gui CaMgO_full_3914.gui
mv CaMgO_full_tmp_3915.f9 CaMgO_full_3915.f9
mv CaMgO_full_tmp_3915.gui CaMgO_full_3915.gui
mv CaMgO_full_tmp_3916.f9 CaMgO_full_3916.f9
mv CaMgO_full_tmp_3916.gui CaMgO_full_3916.gui
mv CaMgO_full_tmp_3917.f9 CaMgO_full_3917.f9
mv CaMgO_full_tmp_3917.gui CaMgO_full_3917.gui
mv CaMgO_full_tmp_3918.f9 CaMgO_full_3918.f9
mv CaMgO_full_tmp_3918.gui CaMgO_full_3918.gui
mv CaMgO_full_tmp_3919.f9 CaMgO_full_3919.f9
mv CaMgO_full_tmp_3919.gui CaMgO_full_3919.gui
mv CaMgO_full_tmp_3920.f9 CaMgO_full_3920.f9
mv CaMgO_full_tmp_3920.gui CaMgO_full_3920.gui
mv CaMgO_full_tmp_3921.f9 CaMgO_full_3921.f9
mv CaMgO_full_tmp_3921.gui CaMgO_full_3921.gui
mv CaMgO_full_tmp_3922.f9 CaMgO_full_3922.f9
mv CaMgO_full_tmp_3922.gui CaMgO_full_3922.gui
mv CaMgO_full_tmp_3923.f9 CaMgO_full_3923.f9
mv CaMgO_full_tmp_3923.gui CaMgO_full_3923.gui
mv CaMgO_full_tmp_3924.f9 CaMgO_full_3924.f9
mv CaMgO_full_tmp_3924.gui CaMgO_full_3924.gui
mv CaMgO_full_tmp_3925.f9 CaMgO_full_3925.f9
mv CaMgO_full_tmp_3925.gui CaMgO_full_3925.gui
mv CaMgO_full_tmp_3926.f9 CaMgO_full_3926.f9
mv CaMgO_full_tmp_3926.gui CaMgO_full_3926.gui
mv CaMgO_full_tmp_3927.f9 CaMgO_full_3927.f9
mv CaMgO_full_tmp_3927.gui CaMgO_full_3927.gui
mv CaMgO_full_tmp_3928.f9 CaMgO_full_3928.f9
mv CaMgO_full_tmp_3928.gui CaMgO_full_3928.gui
mv CaMgO_full_tmp_3929.f9 CaMgO_full_3929.f9
mv CaMgO_full_tmp_3929.gui CaMgO_full_3929.gui
mv CaMgO_full_tmp_3930.f9 CaMgO_full_3930.f9
mv CaMgO_full_tmp_3930.gui CaMgO_full_3930.gui
mv CaMgO_full_tmp_3931.f9 CaMgO_full_3931.f9
mv CaMgO_full_tmp_3931.gui CaMgO_full_3931.gui
mv CaMgO_full_tmp_3932.f9 CaMgO_full_3932.f9
mv CaMgO_full_tmp_3932.gui CaMgO_full_3932.gui
mv CaMgO_full_tmp_3933.f9 CaMgO_full_3933.f9
mv CaMgO_full_tmp_3933.gui CaMgO_full_3933.gui
mv CaMgO_full_tmp_3934.f9 CaMgO_full_3934.f9
mv CaMgO_full_tmp_3934.gui CaMgO_full_3934.gui
mv CaMgO_full_tmp_3935.f9 CaMgO_full_3935.f9
mv CaMgO_full_tmp_3935.gui CaMgO_full_3935.gui
mv CaMgO_full_tmp_3936.f9 CaMgO_full_3936.f9
mv CaMgO_full_tmp_3936.gui CaMgO_full_3936.gui
mv CaMgO_full_tmp_3937.f9 CaMgO_full_3937.f9
mv CaMgO_full_tmp_3937.gui CaMgO_full_3937.gui
mv CaMgO_full_tmp_3938.f9 CaMgO_full_3938.f9
mv CaMgO_full_tmp_3938.gui CaMgO_full_3938.gui
mv CaMgO_full_tmp_3939.f9 CaMgO_full_3939.f9
mv CaMgO_full_tmp_3939.gui CaMgO_full_3939.gui
mv CaMgO_full_tmp_3940.f9 CaMgO_full_3940.f9
mv CaMgO_full_tmp_3940.gui CaMgO_full_3940.gui
mv CaMgO_full_tmp_3941.f9 CaMgO_full_3941.f9
mv CaMgO_full_tmp_3941.gui CaMgO_full_3941.gui
mv CaMgO_full_tmp_3942.f9 CaMgO_full_3942.f9
mv CaMgO_full_tmp_3942.gui CaMgO_full_3942.gui
mv CaMgO_full_tmp_3943.f9 CaMgO_full_3943.f9
mv CaMgO_full_tmp_3943.gui CaMgO_full_3943.gui
mv CaMgO_full_tmp_3944.f9 CaMgO_full_3944.f9
mv CaMgO_full_tmp_3944.gui CaMgO_full_3944.gui
mv CaMgO_full_tmp_3945.f9 CaMgO_full_3945.f9
mv CaMgO_full_tmp_3945.gui CaMgO_full_3945.gui
mv CaMgO_full_tmp_3946.f9 CaMgO_full_3946.f9
mv CaMgO_full_tmp_3946.gui CaMgO_full_3946.gui
mv CaMgO_full_tmp_3947.f9 CaMgO_full_3947.f9
mv CaMgO_full_tmp_3947.gui CaMgO_full_3947.gui
mv CaMgO_full_tmp_3948.f9 CaMgO_full_3948.f9
mv CaMgO_full_tmp_3948.gui CaMgO_full_3948.gui
mv CaMgO_full_tmp_3949.f9 CaMgO_full_3949.f9
mv CaMgO_full_tmp_3949.gui CaMgO_full_3949.gui
mv CaMgO_full_tmp_3950.f9 CaMgO_full_3950.f9
mv CaMgO_full_tmp_3950.gui CaMgO_full_3950.gui
mv CaMgO_full_tmp_3951.f9 CaMgO_full_3951.f9
mv CaMgO_full_tmp_3951.gui CaMgO_full_3951.gui
mv CaMgO_full_tmp_3952.f9 CaMgO_full_3952.f9
mv CaMgO_full_tmp_3952.gui CaMgO_full_3952.gui
mv CaMgO_full_tmp_3953.f9 CaMgO_full_3953.f9
mv CaMgO_full_tmp_3953.gui CaMgO_full_3953.gui
mv CaMgO_full_tmp_3954.f9 CaMgO_full_3954.f9
mv CaMgO_full_tmp_3954.gui CaMgO_full_3954.gui
mv CaMgO_full_tmp_3955.f9 CaMgO_full_3955.f9
mv CaMgO_full_tmp_3955.gui CaMgO_full_3955.gui
mv CaMgO_full_tmp_3956.f9 CaMgO_full_3956.f9
mv CaMgO_full_tmp_3956.gui CaMgO_full_3956.gui
mv CaMgO_full_tmp_3957.f9 CaMgO_full_3957.f9
mv CaMgO_full_tmp_3957.gui CaMgO_full_3957.gui
mv CaMgO_full_tmp_3958.f9 CaMgO_full_3958.f9
mv CaMgO_full_tmp_3958.gui CaMgO_full_3958.gui
mv CaMgO_full_tmp_3959.f9 CaMgO_full_3959.f9
mv CaMgO_full_tmp_3959.gui CaMgO_full_3959.gui
mv CaMgO_full_tmp_3960.f9 CaMgO_full_3960.f9
mv CaMgO_full_tmp_3960.gui CaMgO_full_3960.gui
mv CaMgO_full_tmp_3961.f9 CaMgO_full_3961.f9
mv CaMgO_full_tmp_3961.gui CaMgO_full_3961.gui
mv CaMgO_full_tmp_3962.f9 CaMgO_full_3962.f9
mv CaMgO_full_tmp_3962.gui CaMgO_full_3962.gui
mv CaMgO_full_tmp_3963.f9 CaMgO_full_3963.f9
mv CaMgO_full_tmp_3963.gui CaMgO_full_3963.gui
mv CaMgO_full_tmp_3964.f9 CaMgO_full_3964.f9
mv CaMgO_full_tmp_3964.gui CaMgO_full_3964.gui
mv CaMgO_full_tmp_3965.f9 CaMgO_full_3965.f9
mv CaMgO_full_tmp_3965.gui CaMgO_full_3965.gui
mv CaMgO_full_tmp_3966.f9 CaMgO_full_3966.f9
mv CaMgO_full_tmp_3966.gui CaMgO_full_3966.gui
mv CaMgO_full_tmp_3967.f9 CaMgO_full_3967.f9
mv CaMgO_full_tmp_3967.gui CaMgO_full_3967.gui
mv CaMgO_full_tmp_3968.f9 CaMgO_full_3968.f9
mv CaMgO_full_tmp_3968.gui CaMgO_full_3968.gui
mv CaMgO_full_tmp_3969.f9 CaMgO_full_3969.f9
mv CaMgO_full_tmp_3969.gui CaMgO_full_3969.gui
mv CaMgO_full_tmp_3970.f9 CaMgO_full_3970.f9
mv CaMgO_full_tmp_3970.gui CaMgO_full_3970.gui
mv CaMgO_full_tmp_3971.f9 CaMgO_full_3971.f9
mv CaMgO_full_tmp_3971.gui CaMgO_full_3971.gui
mv CaMgO_full_tmp_3972.f9 CaMgO_full_3972.f9
mv CaMgO_full_tmp_3972.gui CaMgO_full_3972.gui
mv CaMgO_full_tmp_3973.f9 CaMgO_full_3973.f9
mv CaMgO_full_tmp_3973.gui CaMgO_full_3973.gui
mv CaMgO_full_tmp_3974.f9 CaMgO_full_3974.f9
mv CaMgO_full_tmp_3974.gui CaMgO_full_3974.gui
mv CaMgO_full_tmp_3975.f9 CaMgO_full_3975.f9
mv CaMgO_full_tmp_3975.gui CaMgO_full_3975.gui
mv CaMgO_full_tmp_3976.f9 CaMgO_full_3976.f9
mv CaMgO_full_tmp_3976.gui CaMgO_full_3976.gui
mv CaMgO_full_tmp_3977.f9 CaMgO_full_3977.f9
mv CaMgO_full_tmp_3977.gui CaMgO_full_3977.gui
mv CaMgO_full_tmp_3978.f9 CaMgO_full_3978.f9
mv CaMgO_full_tmp_3978.gui CaMgO_full_3978.gui
mv CaMgO_full_tmp_3979.f9 CaMgO_full_3979.f9
mv CaMgO_full_tmp_3979.gui CaMgO_full_3979.gui
mv CaMgO_full_tmp_3980.f9 CaMgO_full_3980.f9
mv CaMgO_full_tmp_3980.gui CaMgO_full_3980.gui
mv CaMgO_full_tmp_3981.f9 CaMgO_full_3981.f9
mv CaMgO_full_tmp_3981.gui CaMgO_full_3981.gui
mv CaMgO_full_tmp_3982.f9 CaMgO_full_3982.f9
mv CaMgO_full_tmp_3982.gui CaMgO_full_3982.gui
mv CaMgO_full_tmp_3983.f9 CaMgO_full_3983.f9
mv CaMgO_full_tmp_3983.gui CaMgO_full_3983.gui
mv CaMgO_full_tmp_3984.f9 CaMgO_full_3984.f9
mv CaMgO_full_tmp_3984.gui CaMgO_full_3984.gui
mv CaMgO_full_tmp_3985.f9 CaMgO_full_3985.f9
mv CaMgO_full_tmp_3985.gui CaMgO_full_3985.gui
mv CaMgO_full_tmp_3986.f9 CaMgO_full_3986.f9
mv CaMgO_full_tmp_3986.gui CaMgO_full_3986.gui
mv CaMgO_full_tmp_3987.f9 CaMgO_full_3987.f9
mv CaMgO_full_tmp_3987.gui CaMgO_full_3987.gui
mv CaMgO_full_tmp_3988.f9 CaMgO_full_3988.f9
mv CaMgO_full_tmp_3988.gui CaMgO_full_3988.gui
mv CaMgO_full_tmp_3989.f9 CaMgO_full_3989.f9
mv CaMgO_full_tmp_3989.gui CaMgO_full_3989.gui
mv CaMgO_full_tmp_3990.f9 CaMgO_full_3990.f9
mv CaMgO_full_tmp_3990.gui CaMgO_full_3990.gui
mv CaMgO_full_tmp_3991.f9 CaMgO_full_3991.f9
mv CaMgO_full_tmp_3991.gui CaMgO_full_3991.gui
mv CaMgO_full_tmp_3992.f9 CaMgO_full_3992.f9
mv CaMgO_full_tmp_3992.gui CaMgO_full_3992.gui
mv CaMgO_full_tmp_3993.f9 CaMgO_full_3993.f9
mv CaMgO_full_tmp_3993.gui CaMgO_full_3993.gui
mv CaMgO_full_tmp_3994.f9 CaMgO_full_3994.f9
mv CaMgO_full_tmp_3994.gui CaMgO_full_3994.gui
mv CaMgO_full_tmp_3995.f9 CaMgO_full_3995.f9
mv CaMgO_full_tmp_3995.gui CaMgO_full_3995.gui
mv CaMgO_full_tmp_3996.f9 CaMgO_full_3996.f9
mv CaMgO_full_tmp_3996.gui CaMgO_full_3996.gui
mv CaMgO_full_tmp_3997.f9 CaMgO_full_3997.f9
mv CaMgO_full_tmp_3997.gui CaMgO_full_3997.gui
mv CaMgO_full_tmp_3998.f9 CaMgO_full_3998.f9
mv CaMgO_full_tmp_3998.gui CaMgO_full_3998.gui
mv CaMgO_full_tmp_3999.f9 CaMgO_full_3999.f9
mv CaMgO_full_tmp_3999.gui CaMgO_full_3999.gui
