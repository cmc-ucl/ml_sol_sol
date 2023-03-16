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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3001
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3001


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3002
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3002


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3003
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3003


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3004
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3004


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3005
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3005


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3006
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3006


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3007
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3007


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3008
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3008


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3009
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3009


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3010
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3010


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3011
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3011


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3012
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3012


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3013
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3013


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3014
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3014


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3015
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3015


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3016
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3016


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3017
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3017


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3018
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3018


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3019
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3019


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3020
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3020


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3021
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3021


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3022
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3022


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3023
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3023


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3024
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3024


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3025
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3025


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3026
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3026


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3027
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3027


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3028
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3028


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3029
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3029


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3030
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3030


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3031
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3031


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3032
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3032


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3033
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3033


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3034
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3034


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3035
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3035


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3036
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3036


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3037
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3037


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3038
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3038


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3039
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3039


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3040
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3040


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3041
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3041


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3042
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3042


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3043
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3043


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3044
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3044


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3045
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3045


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3046
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3046


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3047
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3047


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3048
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3048


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3049
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3049


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3050
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3050


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3051
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3051


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3052
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3052


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3053
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3053


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3054
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3054


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3055
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3055


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3056
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3056


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3057
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3057


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3058
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3058


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3059
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3059


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3060
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3060


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3061
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3061


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3062
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3062


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3063
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3063


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3064
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3064


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3065
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3065


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3066
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3066


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3067
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3067


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3068
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3068


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3069
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3069


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3070
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3070


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3071
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3071


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3072
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3072


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3073
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3073


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3074
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3074


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3075
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3075


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3076
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3076


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3077
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3077


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3078
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3078


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3079
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3079


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3080
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3080


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3081
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3081


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3082
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3082


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3083
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3083


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3084
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3084


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3085
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3085


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3086
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3086


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3087
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3087


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3088
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3088


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3089
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3089


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3090
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3090


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3091
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3091


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3092
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3092


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3093
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3093


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3094
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3094


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3095
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3095


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3096
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3096


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3097
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3097


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3098
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3098


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3099
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3099


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3100
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3100


