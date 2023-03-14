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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3001 CaMgO_coord_3001
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3001


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3002 CaMgO_coord_3002
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3002


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3003 CaMgO_coord_3003
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3003


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3004 CaMgO_coord_3004
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3004


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3005 CaMgO_coord_3005
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3005


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3006 CaMgO_coord_3006
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3006


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3007 CaMgO_coord_3007
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3007


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3008 CaMgO_coord_3008
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3008


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3009 CaMgO_coord_3009
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3009


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3010 CaMgO_coord_3010
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3010


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3011 CaMgO_coord_3011
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3011


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3012 CaMgO_coord_3012
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3012


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3013 CaMgO_coord_3013
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3013


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3014 CaMgO_coord_3014
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3014


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3015 CaMgO_coord_3015
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3015


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3016 CaMgO_coord_3016
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3016


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3017 CaMgO_coord_3017
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3017


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3018 CaMgO_coord_3018
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3018


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3019 CaMgO_coord_3019
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3019


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3020 CaMgO_coord_3020
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3020


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3021 CaMgO_coord_3021
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3021


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3022 CaMgO_coord_3022
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3022


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3023 CaMgO_coord_3023
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3023


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3024 CaMgO_coord_3024
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3024


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3025 CaMgO_coord_3025
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3025


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3026 CaMgO_coord_3026
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3026


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3027 CaMgO_coord_3027
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3027


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3028 CaMgO_coord_3028
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3028


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3029 CaMgO_coord_3029
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3029


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3030 CaMgO_coord_3030
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3030


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3031 CaMgO_coord_3031
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3031


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3032 CaMgO_coord_3032
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3032


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3033 CaMgO_coord_3033
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3033


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3034 CaMgO_coord_3034
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3034


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3035 CaMgO_coord_3035
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3035


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3036 CaMgO_coord_3036
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3036


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3037 CaMgO_coord_3037
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3037


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3038 CaMgO_coord_3038
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3038


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3039 CaMgO_coord_3039
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3039


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3040 CaMgO_coord_3040
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3040


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3041 CaMgO_coord_3041
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3041


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3042 CaMgO_coord_3042
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3042


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3043 CaMgO_coord_3043
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3043


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3044 CaMgO_coord_3044
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3044


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3045 CaMgO_coord_3045
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3045


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3046 CaMgO_coord_3046
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3046


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3047 CaMgO_coord_3047
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3047


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3048 CaMgO_coord_3048
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3048


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3049 CaMgO_coord_3049
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3049


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3050 CaMgO_coord_3050
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3050


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3051 CaMgO_coord_3051
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3051


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3052 CaMgO_coord_3052
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3052


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3053 CaMgO_coord_3053
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3053


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3054 CaMgO_coord_3054
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3054


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3055 CaMgO_coord_3055
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3055


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3056 CaMgO_coord_3056
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3056


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3057 CaMgO_coord_3057
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3057


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3058 CaMgO_coord_3058
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3058


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3059 CaMgO_coord_3059
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3059


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3060 CaMgO_coord_3060
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3060


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3061 CaMgO_coord_3061
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3061


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3062 CaMgO_coord_3062
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3062


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3063 CaMgO_coord_3063
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3063


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3064 CaMgO_coord_3064
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3064


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3065 CaMgO_coord_3065
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3065


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3066 CaMgO_coord_3066
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3066


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3067 CaMgO_coord_3067
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3067


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3068 CaMgO_coord_3068
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3068


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3069 CaMgO_coord_3069
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3069


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3070 CaMgO_coord_3070
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3070


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3071 CaMgO_coord_3071
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3071


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3072 CaMgO_coord_3072
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3072


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3073 CaMgO_coord_3073
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3073


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3074 CaMgO_coord_3074
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3074


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3075 CaMgO_coord_3075
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3075


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3076 CaMgO_coord_3076
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3076


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3077 CaMgO_coord_3077
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3077


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3078 CaMgO_coord_3078
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3078


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3079 CaMgO_coord_3079
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3079


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3080 CaMgO_coord_3080
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3080


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3081 CaMgO_coord_3081
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3081


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3082 CaMgO_coord_3082
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3082


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3083 CaMgO_coord_3083
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3083


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3084 CaMgO_coord_3084
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3084


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3085 CaMgO_coord_3085
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3085


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3086 CaMgO_coord_3086
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3086


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3087 CaMgO_coord_3087
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3087


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3088 CaMgO_coord_3088
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3088


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3089 CaMgO_coord_3089
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3089


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3090 CaMgO_coord_3090
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3090


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3091 CaMgO_coord_3091
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3091


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3092 CaMgO_coord_3092
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3092


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3093 CaMgO_coord_3093
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3093


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3094 CaMgO_coord_3094
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3094


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3095 CaMgO_coord_3095
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3095


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3096 CaMgO_coord_3096
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3096


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3097 CaMgO_coord_3097
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3097


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3098 CaMgO_coord_3098
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3098


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3099 CaMgO_coord_3099
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3099


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3100 CaMgO_coord_3100
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3100


