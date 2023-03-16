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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3101
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3101


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3102
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3102


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3103
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3103


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3104
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3104


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3105
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3105


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3106
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3106


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3107
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3107


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3108
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3108


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3109
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3109


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3110
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3110


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3111
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3111


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3112
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3112


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3113
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3113


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3114
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3114


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3115
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3115


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3116
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3116


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3117
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3117


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3118
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3118


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3119
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3119


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3120
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3120


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3121
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3121


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3122
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3122


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3123
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3123


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3124
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3124


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3125
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3125


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3126
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3126


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3127
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3127


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3128
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3128


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3129
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3129


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3130
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3130


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3131
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3131


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3132
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3132


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3133
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3133


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3134
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3134


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3135
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3135


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3136
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3136


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3137
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3137


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3138
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3138


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3139
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3139


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3140
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3140


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3141
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3141


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3142
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3142


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3143
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3143


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3144
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3144


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3145
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3145


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3146
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3146


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3147
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3147


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3148
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3148


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3149
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3149


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3150
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3150


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3151
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3151


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3152
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3152


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3153
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3153


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3154
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3154


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3155
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3155


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3156
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3156


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3157
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3157


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3158
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3158


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3159
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3159


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3160
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3160


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3161
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3161


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3162
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3162


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3163
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3163


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3164
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3164


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3165
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3165


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3166
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3166


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3167
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3167


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3168
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3168


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3169
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3169


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3170
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3170


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3171
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3171


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3172
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3172


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3173
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3173


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3174
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3174


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3175
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3175


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3176
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3176


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3177
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3177


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3178
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3178


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3179
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3179


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3180
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3180


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3181
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3181


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3182
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3182


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3183
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3183


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3184
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3184


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3185
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3185


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3186
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3186


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3187
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3187


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3188
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3188


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3189
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3189


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3190
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3190


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3191
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3191


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3192
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3192


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3193
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3193


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3194
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3194


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3195
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3195


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3196
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3196


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3197
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3197


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3198
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3198


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3199
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3199


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3200


