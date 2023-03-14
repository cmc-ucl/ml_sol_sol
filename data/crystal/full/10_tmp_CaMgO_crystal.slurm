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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_900 CaMgO_full_tmp_900 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_901 CaMgO_full_tmp_901 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_902 CaMgO_full_tmp_902 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_903 CaMgO_full_tmp_903 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_904 CaMgO_full_tmp_904 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_905 CaMgO_full_tmp_905 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_906 CaMgO_full_tmp_906 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_907 CaMgO_full_tmp_907 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_908 CaMgO_full_tmp_908 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_909 CaMgO_full_tmp_909 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_910 CaMgO_full_tmp_910 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_911 CaMgO_full_tmp_911 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_912 CaMgO_full_tmp_912 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_913 CaMgO_full_tmp_913 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_914 CaMgO_full_tmp_914 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_915 CaMgO_full_tmp_915 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_916 CaMgO_full_tmp_916 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_917 CaMgO_full_tmp_917 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_918 CaMgO_full_tmp_918 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_919 CaMgO_full_tmp_919 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_920 CaMgO_full_tmp_920 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_921 CaMgO_full_tmp_921 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_922 CaMgO_full_tmp_922 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_923 CaMgO_full_tmp_923 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_924 CaMgO_full_tmp_924 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_925 CaMgO_full_tmp_925 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_926 CaMgO_full_tmp_926 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_927 CaMgO_full_tmp_927 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_928 CaMgO_full_tmp_928 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_929 CaMgO_full_tmp_929 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_930 CaMgO_full_tmp_930 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_931 CaMgO_full_tmp_931 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_932 CaMgO_full_tmp_932 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_933 CaMgO_full_tmp_933 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_934 CaMgO_full_tmp_934 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_935 CaMgO_full_tmp_935 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_936 CaMgO_full_tmp_936 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_937 CaMgO_full_tmp_937 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_938 CaMgO_full_tmp_938 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_939 CaMgO_full_tmp_939 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_940 CaMgO_full_tmp_940 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_941 CaMgO_full_tmp_941 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_942 CaMgO_full_tmp_942 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_943 CaMgO_full_tmp_943 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_944 CaMgO_full_tmp_944 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_945 CaMgO_full_tmp_945 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_946 CaMgO_full_tmp_946 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_947 CaMgO_full_tmp_947 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_948 CaMgO_full_tmp_948 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_949 CaMgO_full_tmp_949 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_950 CaMgO_full_tmp_950 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_951 CaMgO_full_tmp_951 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_952 CaMgO_full_tmp_952 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_953 CaMgO_full_tmp_953 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_954 CaMgO_full_tmp_954 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_955 CaMgO_full_tmp_955 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_956 CaMgO_full_tmp_956 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_957 CaMgO_full_tmp_957 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_958 CaMgO_full_tmp_958 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_959 CaMgO_full_tmp_959 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_960 CaMgO_full_tmp_960 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_961 CaMgO_full_tmp_961 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_962 CaMgO_full_tmp_962 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_963 CaMgO_full_tmp_963 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_964 CaMgO_full_tmp_964 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_965 CaMgO_full_tmp_965 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_966 CaMgO_full_tmp_966 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_967 CaMgO_full_tmp_967 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_968 CaMgO_full_tmp_968 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_969 CaMgO_full_tmp_969 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_970 CaMgO_full_tmp_970 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_971 CaMgO_full_tmp_971 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_972 CaMgO_full_tmp_972 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_973 CaMgO_full_tmp_973 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_974 CaMgO_full_tmp_974 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_975 CaMgO_full_tmp_975 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_976 CaMgO_full_tmp_976 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_977 CaMgO_full_tmp_977 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_978 CaMgO_full_tmp_978 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_979 CaMgO_full_tmp_979 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_980 CaMgO_full_tmp_980 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_981 CaMgO_full_tmp_981 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_982 CaMgO_full_tmp_982 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_983 CaMgO_full_tmp_983 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_984 CaMgO_full_tmp_984 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_985 CaMgO_full_tmp_985 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_986 CaMgO_full_tmp_986 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_987 CaMgO_full_tmp_987 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_988 CaMgO_full_tmp_988 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_989 CaMgO_full_tmp_989 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_990 CaMgO_full_tmp_990 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_991 CaMgO_full_tmp_991 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_992 CaMgO_full_tmp_992 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_993 CaMgO_full_tmp_993 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_994 CaMgO_full_tmp_994 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_995 CaMgO_full_tmp_995 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_996 CaMgO_full_tmp_996 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_997 CaMgO_full_tmp_997 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_998 CaMgO_full_tmp_998 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_999 CaMgO_full_tmp_999 &

wait
mv CaMgO_full_tmp_900.f9 CaMgO_full_900.f9
mv CaMgO_full_tmp_900.gui CaMgO_full_900.gui
mv CaMgO_full_tmp_901.f9 CaMgO_full_901.f9
mv CaMgO_full_tmp_901.gui CaMgO_full_901.gui
mv CaMgO_full_tmp_902.f9 CaMgO_full_902.f9
mv CaMgO_full_tmp_902.gui CaMgO_full_902.gui
mv CaMgO_full_tmp_903.f9 CaMgO_full_903.f9
mv CaMgO_full_tmp_903.gui CaMgO_full_903.gui
mv CaMgO_full_tmp_904.f9 CaMgO_full_904.f9
mv CaMgO_full_tmp_904.gui CaMgO_full_904.gui
mv CaMgO_full_tmp_905.f9 CaMgO_full_905.f9
mv CaMgO_full_tmp_905.gui CaMgO_full_905.gui
mv CaMgO_full_tmp_906.f9 CaMgO_full_906.f9
mv CaMgO_full_tmp_906.gui CaMgO_full_906.gui
mv CaMgO_full_tmp_907.f9 CaMgO_full_907.f9
mv CaMgO_full_tmp_907.gui CaMgO_full_907.gui
mv CaMgO_full_tmp_908.f9 CaMgO_full_908.f9
mv CaMgO_full_tmp_908.gui CaMgO_full_908.gui
mv CaMgO_full_tmp_909.f9 CaMgO_full_909.f9
mv CaMgO_full_tmp_909.gui CaMgO_full_909.gui
mv CaMgO_full_tmp_910.f9 CaMgO_full_910.f9
mv CaMgO_full_tmp_910.gui CaMgO_full_910.gui
mv CaMgO_full_tmp_911.f9 CaMgO_full_911.f9
mv CaMgO_full_tmp_911.gui CaMgO_full_911.gui
mv CaMgO_full_tmp_912.f9 CaMgO_full_912.f9
mv CaMgO_full_tmp_912.gui CaMgO_full_912.gui
mv CaMgO_full_tmp_913.f9 CaMgO_full_913.f9
mv CaMgO_full_tmp_913.gui CaMgO_full_913.gui
mv CaMgO_full_tmp_914.f9 CaMgO_full_914.f9
mv CaMgO_full_tmp_914.gui CaMgO_full_914.gui
mv CaMgO_full_tmp_915.f9 CaMgO_full_915.f9
mv CaMgO_full_tmp_915.gui CaMgO_full_915.gui
mv CaMgO_full_tmp_916.f9 CaMgO_full_916.f9
mv CaMgO_full_tmp_916.gui CaMgO_full_916.gui
mv CaMgO_full_tmp_917.f9 CaMgO_full_917.f9
mv CaMgO_full_tmp_917.gui CaMgO_full_917.gui
mv CaMgO_full_tmp_918.f9 CaMgO_full_918.f9
mv CaMgO_full_tmp_918.gui CaMgO_full_918.gui
mv CaMgO_full_tmp_919.f9 CaMgO_full_919.f9
mv CaMgO_full_tmp_919.gui CaMgO_full_919.gui
mv CaMgO_full_tmp_920.f9 CaMgO_full_920.f9
mv CaMgO_full_tmp_920.gui CaMgO_full_920.gui
mv CaMgO_full_tmp_921.f9 CaMgO_full_921.f9
mv CaMgO_full_tmp_921.gui CaMgO_full_921.gui
mv CaMgO_full_tmp_922.f9 CaMgO_full_922.f9
mv CaMgO_full_tmp_922.gui CaMgO_full_922.gui
mv CaMgO_full_tmp_923.f9 CaMgO_full_923.f9
mv CaMgO_full_tmp_923.gui CaMgO_full_923.gui
mv CaMgO_full_tmp_924.f9 CaMgO_full_924.f9
mv CaMgO_full_tmp_924.gui CaMgO_full_924.gui
mv CaMgO_full_tmp_925.f9 CaMgO_full_925.f9
mv CaMgO_full_tmp_925.gui CaMgO_full_925.gui
mv CaMgO_full_tmp_926.f9 CaMgO_full_926.f9
mv CaMgO_full_tmp_926.gui CaMgO_full_926.gui
mv CaMgO_full_tmp_927.f9 CaMgO_full_927.f9
mv CaMgO_full_tmp_927.gui CaMgO_full_927.gui
mv CaMgO_full_tmp_928.f9 CaMgO_full_928.f9
mv CaMgO_full_tmp_928.gui CaMgO_full_928.gui
mv CaMgO_full_tmp_929.f9 CaMgO_full_929.f9
mv CaMgO_full_tmp_929.gui CaMgO_full_929.gui
mv CaMgO_full_tmp_930.f9 CaMgO_full_930.f9
mv CaMgO_full_tmp_930.gui CaMgO_full_930.gui
mv CaMgO_full_tmp_931.f9 CaMgO_full_931.f9
mv CaMgO_full_tmp_931.gui CaMgO_full_931.gui
mv CaMgO_full_tmp_932.f9 CaMgO_full_932.f9
mv CaMgO_full_tmp_932.gui CaMgO_full_932.gui
mv CaMgO_full_tmp_933.f9 CaMgO_full_933.f9
mv CaMgO_full_tmp_933.gui CaMgO_full_933.gui
mv CaMgO_full_tmp_934.f9 CaMgO_full_934.f9
mv CaMgO_full_tmp_934.gui CaMgO_full_934.gui
mv CaMgO_full_tmp_935.f9 CaMgO_full_935.f9
mv CaMgO_full_tmp_935.gui CaMgO_full_935.gui
mv CaMgO_full_tmp_936.f9 CaMgO_full_936.f9
mv CaMgO_full_tmp_936.gui CaMgO_full_936.gui
mv CaMgO_full_tmp_937.f9 CaMgO_full_937.f9
mv CaMgO_full_tmp_937.gui CaMgO_full_937.gui
mv CaMgO_full_tmp_938.f9 CaMgO_full_938.f9
mv CaMgO_full_tmp_938.gui CaMgO_full_938.gui
mv CaMgO_full_tmp_939.f9 CaMgO_full_939.f9
mv CaMgO_full_tmp_939.gui CaMgO_full_939.gui
mv CaMgO_full_tmp_940.f9 CaMgO_full_940.f9
mv CaMgO_full_tmp_940.gui CaMgO_full_940.gui
mv CaMgO_full_tmp_941.f9 CaMgO_full_941.f9
mv CaMgO_full_tmp_941.gui CaMgO_full_941.gui
mv CaMgO_full_tmp_942.f9 CaMgO_full_942.f9
mv CaMgO_full_tmp_942.gui CaMgO_full_942.gui
mv CaMgO_full_tmp_943.f9 CaMgO_full_943.f9
mv CaMgO_full_tmp_943.gui CaMgO_full_943.gui
mv CaMgO_full_tmp_944.f9 CaMgO_full_944.f9
mv CaMgO_full_tmp_944.gui CaMgO_full_944.gui
mv CaMgO_full_tmp_945.f9 CaMgO_full_945.f9
mv CaMgO_full_tmp_945.gui CaMgO_full_945.gui
mv CaMgO_full_tmp_946.f9 CaMgO_full_946.f9
mv CaMgO_full_tmp_946.gui CaMgO_full_946.gui
mv CaMgO_full_tmp_947.f9 CaMgO_full_947.f9
mv CaMgO_full_tmp_947.gui CaMgO_full_947.gui
mv CaMgO_full_tmp_948.f9 CaMgO_full_948.f9
mv CaMgO_full_tmp_948.gui CaMgO_full_948.gui
mv CaMgO_full_tmp_949.f9 CaMgO_full_949.f9
mv CaMgO_full_tmp_949.gui CaMgO_full_949.gui
mv CaMgO_full_tmp_950.f9 CaMgO_full_950.f9
mv CaMgO_full_tmp_950.gui CaMgO_full_950.gui
mv CaMgO_full_tmp_951.f9 CaMgO_full_951.f9
mv CaMgO_full_tmp_951.gui CaMgO_full_951.gui
mv CaMgO_full_tmp_952.f9 CaMgO_full_952.f9
mv CaMgO_full_tmp_952.gui CaMgO_full_952.gui
mv CaMgO_full_tmp_953.f9 CaMgO_full_953.f9
mv CaMgO_full_tmp_953.gui CaMgO_full_953.gui
mv CaMgO_full_tmp_954.f9 CaMgO_full_954.f9
mv CaMgO_full_tmp_954.gui CaMgO_full_954.gui
mv CaMgO_full_tmp_955.f9 CaMgO_full_955.f9
mv CaMgO_full_tmp_955.gui CaMgO_full_955.gui
mv CaMgO_full_tmp_956.f9 CaMgO_full_956.f9
mv CaMgO_full_tmp_956.gui CaMgO_full_956.gui
mv CaMgO_full_tmp_957.f9 CaMgO_full_957.f9
mv CaMgO_full_tmp_957.gui CaMgO_full_957.gui
mv CaMgO_full_tmp_958.f9 CaMgO_full_958.f9
mv CaMgO_full_tmp_958.gui CaMgO_full_958.gui
mv CaMgO_full_tmp_959.f9 CaMgO_full_959.f9
mv CaMgO_full_tmp_959.gui CaMgO_full_959.gui
mv CaMgO_full_tmp_960.f9 CaMgO_full_960.f9
mv CaMgO_full_tmp_960.gui CaMgO_full_960.gui
mv CaMgO_full_tmp_961.f9 CaMgO_full_961.f9
mv CaMgO_full_tmp_961.gui CaMgO_full_961.gui
mv CaMgO_full_tmp_962.f9 CaMgO_full_962.f9
mv CaMgO_full_tmp_962.gui CaMgO_full_962.gui
mv CaMgO_full_tmp_963.f9 CaMgO_full_963.f9
mv CaMgO_full_tmp_963.gui CaMgO_full_963.gui
mv CaMgO_full_tmp_964.f9 CaMgO_full_964.f9
mv CaMgO_full_tmp_964.gui CaMgO_full_964.gui
mv CaMgO_full_tmp_965.f9 CaMgO_full_965.f9
mv CaMgO_full_tmp_965.gui CaMgO_full_965.gui
mv CaMgO_full_tmp_966.f9 CaMgO_full_966.f9
mv CaMgO_full_tmp_966.gui CaMgO_full_966.gui
mv CaMgO_full_tmp_967.f9 CaMgO_full_967.f9
mv CaMgO_full_tmp_967.gui CaMgO_full_967.gui
mv CaMgO_full_tmp_968.f9 CaMgO_full_968.f9
mv CaMgO_full_tmp_968.gui CaMgO_full_968.gui
mv CaMgO_full_tmp_969.f9 CaMgO_full_969.f9
mv CaMgO_full_tmp_969.gui CaMgO_full_969.gui
mv CaMgO_full_tmp_970.f9 CaMgO_full_970.f9
mv CaMgO_full_tmp_970.gui CaMgO_full_970.gui
mv CaMgO_full_tmp_971.f9 CaMgO_full_971.f9
mv CaMgO_full_tmp_971.gui CaMgO_full_971.gui
mv CaMgO_full_tmp_972.f9 CaMgO_full_972.f9
mv CaMgO_full_tmp_972.gui CaMgO_full_972.gui
mv CaMgO_full_tmp_973.f9 CaMgO_full_973.f9
mv CaMgO_full_tmp_973.gui CaMgO_full_973.gui
mv CaMgO_full_tmp_974.f9 CaMgO_full_974.f9
mv CaMgO_full_tmp_974.gui CaMgO_full_974.gui
mv CaMgO_full_tmp_975.f9 CaMgO_full_975.f9
mv CaMgO_full_tmp_975.gui CaMgO_full_975.gui
mv CaMgO_full_tmp_976.f9 CaMgO_full_976.f9
mv CaMgO_full_tmp_976.gui CaMgO_full_976.gui
mv CaMgO_full_tmp_977.f9 CaMgO_full_977.f9
mv CaMgO_full_tmp_977.gui CaMgO_full_977.gui
mv CaMgO_full_tmp_978.f9 CaMgO_full_978.f9
mv CaMgO_full_tmp_978.gui CaMgO_full_978.gui
mv CaMgO_full_tmp_979.f9 CaMgO_full_979.f9
mv CaMgO_full_tmp_979.gui CaMgO_full_979.gui
mv CaMgO_full_tmp_980.f9 CaMgO_full_980.f9
mv CaMgO_full_tmp_980.gui CaMgO_full_980.gui
mv CaMgO_full_tmp_981.f9 CaMgO_full_981.f9
mv CaMgO_full_tmp_981.gui CaMgO_full_981.gui
mv CaMgO_full_tmp_982.f9 CaMgO_full_982.f9
mv CaMgO_full_tmp_982.gui CaMgO_full_982.gui
mv CaMgO_full_tmp_983.f9 CaMgO_full_983.f9
mv CaMgO_full_tmp_983.gui CaMgO_full_983.gui
mv CaMgO_full_tmp_984.f9 CaMgO_full_984.f9
mv CaMgO_full_tmp_984.gui CaMgO_full_984.gui
mv CaMgO_full_tmp_985.f9 CaMgO_full_985.f9
mv CaMgO_full_tmp_985.gui CaMgO_full_985.gui
mv CaMgO_full_tmp_986.f9 CaMgO_full_986.f9
mv CaMgO_full_tmp_986.gui CaMgO_full_986.gui
mv CaMgO_full_tmp_987.f9 CaMgO_full_987.f9
mv CaMgO_full_tmp_987.gui CaMgO_full_987.gui
mv CaMgO_full_tmp_988.f9 CaMgO_full_988.f9
mv CaMgO_full_tmp_988.gui CaMgO_full_988.gui
mv CaMgO_full_tmp_989.f9 CaMgO_full_989.f9
mv CaMgO_full_tmp_989.gui CaMgO_full_989.gui
mv CaMgO_full_tmp_990.f9 CaMgO_full_990.f9
mv CaMgO_full_tmp_990.gui CaMgO_full_990.gui
mv CaMgO_full_tmp_991.f9 CaMgO_full_991.f9
mv CaMgO_full_tmp_991.gui CaMgO_full_991.gui
mv CaMgO_full_tmp_992.f9 CaMgO_full_992.f9
mv CaMgO_full_tmp_992.gui CaMgO_full_992.gui
mv CaMgO_full_tmp_993.f9 CaMgO_full_993.f9
mv CaMgO_full_tmp_993.gui CaMgO_full_993.gui
mv CaMgO_full_tmp_994.f9 CaMgO_full_994.f9
mv CaMgO_full_tmp_994.gui CaMgO_full_994.gui
mv CaMgO_full_tmp_995.f9 CaMgO_full_995.f9
mv CaMgO_full_tmp_995.gui CaMgO_full_995.gui
mv CaMgO_full_tmp_996.f9 CaMgO_full_996.f9
mv CaMgO_full_tmp_996.gui CaMgO_full_996.gui
mv CaMgO_full_tmp_997.f9 CaMgO_full_997.f9
mv CaMgO_full_tmp_997.gui CaMgO_full_997.gui
mv CaMgO_full_tmp_998.f9 CaMgO_full_998.f9
mv CaMgO_full_tmp_998.gui CaMgO_full_998.gui
mv CaMgO_full_tmp_999.f9 CaMgO_full_999.f9
mv CaMgO_full_tmp_999.gui CaMgO_full_999.gui
