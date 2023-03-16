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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2901 CaMgO_coord_2901
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2901


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2902 CaMgO_coord_2902
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2902


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2903 CaMgO_coord_2903
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2903


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2904 CaMgO_coord_2904
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2904


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2905 CaMgO_coord_2905
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2905


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2906 CaMgO_coord_2906
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2906


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2907 CaMgO_coord_2907
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2907


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2908 CaMgO_coord_2908
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2908


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2909 CaMgO_coord_2909
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2909


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2910 CaMgO_coord_2910
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2910


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2911 CaMgO_coord_2911
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2911


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2912 CaMgO_coord_2912
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2912


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2913 CaMgO_coord_2913
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2913


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2914 CaMgO_coord_2914
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2914


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2915 CaMgO_coord_2915
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2915


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2916 CaMgO_coord_2916
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2916


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2917 CaMgO_coord_2917
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2917


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2918 CaMgO_coord_2918
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2918


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2919 CaMgO_coord_2919
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2919


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2920 CaMgO_coord_2920
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2920


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2921 CaMgO_coord_2921
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2921


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2922 CaMgO_coord_2922
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2922


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2923 CaMgO_coord_2923
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2923


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2924 CaMgO_coord_2924
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2924


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2925 CaMgO_coord_2925
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2925


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2926 CaMgO_coord_2926
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2926


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2927 CaMgO_coord_2927
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2927


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2928 CaMgO_coord_2928
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2928


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2929 CaMgO_coord_2929
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2929


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2930 CaMgO_coord_2930
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2930


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2931 CaMgO_coord_2931
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2931


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2932 CaMgO_coord_2932
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2932


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2933 CaMgO_coord_2933
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2933


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2934 CaMgO_coord_2934
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2934


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2935 CaMgO_coord_2935
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2935


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2936 CaMgO_coord_2936
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2936


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2937 CaMgO_coord_2937
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2937


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2938 CaMgO_coord_2938
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2938


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2939 CaMgO_coord_2939
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2939


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2940 CaMgO_coord_2940
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2940


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2941 CaMgO_coord_2941
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2941


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2942 CaMgO_coord_2942
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2942


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2943 CaMgO_coord_2943
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2943


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2944 CaMgO_coord_2944
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2944


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2945 CaMgO_coord_2945
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2945


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2946 CaMgO_coord_2946
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2946


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2947 CaMgO_coord_2947
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2947


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2948 CaMgO_coord_2948
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2948


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2949 CaMgO_coord_2949
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2949


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2950 CaMgO_coord_2950
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2950


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2951 CaMgO_coord_2951
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2951


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2952 CaMgO_coord_2952
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2952


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2953 CaMgO_coord_2953
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2953


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2954 CaMgO_coord_2954
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2954


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2955 CaMgO_coord_2955
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2955


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2956 CaMgO_coord_2956
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2956


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2957 CaMgO_coord_2957
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2957


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2958 CaMgO_coord_2958
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2958


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2959 CaMgO_coord_2959
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2959


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2960 CaMgO_coord_2960
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2960


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2961 CaMgO_coord_2961
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2961


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2962 CaMgO_coord_2962
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2962


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2963 CaMgO_coord_2963
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2963


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2964 CaMgO_coord_2964
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2964


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2965 CaMgO_coord_2965
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2965


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2966 CaMgO_coord_2966
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2966


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2967 CaMgO_coord_2967
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2967


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2968 CaMgO_coord_2968
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2968


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2969 CaMgO_coord_2969
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2969


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2970 CaMgO_coord_2970
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2970


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2971 CaMgO_coord_2971
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2971


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2972 CaMgO_coord_2972
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2972


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2973 CaMgO_coord_2973
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2973


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2974 CaMgO_coord_2974
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2974


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2975 CaMgO_coord_2975
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2975


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2976 CaMgO_coord_2976
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2976


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2977 CaMgO_coord_2977
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2977


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2978 CaMgO_coord_2978
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2978


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2979 CaMgO_coord_2979
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2979


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2980 CaMgO_coord_2980
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2980


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2981 CaMgO_coord_2981
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2981


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2982 CaMgO_coord_2982
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2982


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2983 CaMgO_coord_2983
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2983


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2984 CaMgO_coord_2984
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2984


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2985 CaMgO_coord_2985
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2985


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2986 CaMgO_coord_2986
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2986


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2987 CaMgO_coord_2987
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2987


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2988 CaMgO_coord_2988
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2988


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2989 CaMgO_coord_2989
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2989


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2990 CaMgO_coord_2990
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2990


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2991 CaMgO_coord_2991
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2991


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2992 CaMgO_coord_2992
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2992


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2993 CaMgO_coord_2993
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2993


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2994 CaMgO_coord_2994
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2994


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2995 CaMgO_coord_2995
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2995


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2996 CaMgO_coord_2996
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2996


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2997 CaMgO_coord_2997
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2997


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2998 CaMgO_coord_2998
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2998


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2999 CaMgO_coord_2999
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2999


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3000 CaMgO_coord_3000
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3000


