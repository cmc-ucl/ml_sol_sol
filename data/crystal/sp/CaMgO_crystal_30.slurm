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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2999


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3000
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3000


