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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3101 CaMgO_coord_3101
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3101


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3102 CaMgO_coord_3102
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3102


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3103 CaMgO_coord_3103
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3103


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3104 CaMgO_coord_3104
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3104


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3105 CaMgO_coord_3105
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3105


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3106 CaMgO_coord_3106
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3106


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3107 CaMgO_coord_3107
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3107


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3108 CaMgO_coord_3108
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3108


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3109 CaMgO_coord_3109
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3109


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3110 CaMgO_coord_3110
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3110


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3111 CaMgO_coord_3111
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3111


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3112 CaMgO_coord_3112
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3112


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3113 CaMgO_coord_3113
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3113


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3114 CaMgO_coord_3114
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3114


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3115 CaMgO_coord_3115
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3115


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3116 CaMgO_coord_3116
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3116


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3117 CaMgO_coord_3117
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3117


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3118 CaMgO_coord_3118
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3118


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3119 CaMgO_coord_3119
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3119


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3120 CaMgO_coord_3120
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3120


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3121 CaMgO_coord_3121
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3121


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3122 CaMgO_coord_3122
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3122


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3123 CaMgO_coord_3123
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3123


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3124 CaMgO_coord_3124
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3124


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3125 CaMgO_coord_3125
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3125


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3126 CaMgO_coord_3126
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3126


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3127 CaMgO_coord_3127
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3127


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3128 CaMgO_coord_3128
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3128


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3129 CaMgO_coord_3129
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3129


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3130 CaMgO_coord_3130
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3130


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3131 CaMgO_coord_3131
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3131


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3132 CaMgO_coord_3132
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3132


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3133 CaMgO_coord_3133
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3133


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3134 CaMgO_coord_3134
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3134


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3135 CaMgO_coord_3135
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3135


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3136 CaMgO_coord_3136
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3136


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3137 CaMgO_coord_3137
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3137


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3138 CaMgO_coord_3138
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3138


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3139 CaMgO_coord_3139
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3139


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3140 CaMgO_coord_3140
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3140


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3141 CaMgO_coord_3141
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3141


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3142 CaMgO_coord_3142
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3142


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3143 CaMgO_coord_3143
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3143


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3144 CaMgO_coord_3144
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3144


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3145 CaMgO_coord_3145
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3145


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3146 CaMgO_coord_3146
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3146


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3147 CaMgO_coord_3147
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3147


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3148 CaMgO_coord_3148
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3148


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3149 CaMgO_coord_3149
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3149


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3150 CaMgO_coord_3150
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3150


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3151 CaMgO_coord_3151
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3151


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3152 CaMgO_coord_3152
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3152


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3153 CaMgO_coord_3153
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3153


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3154 CaMgO_coord_3154
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3154


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3155 CaMgO_coord_3155
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3155


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3156 CaMgO_coord_3156
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3156


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3157 CaMgO_coord_3157
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3157


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3158 CaMgO_coord_3158
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3158


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3159 CaMgO_coord_3159
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3159


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3160 CaMgO_coord_3160
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3160


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3161 CaMgO_coord_3161
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3161


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3162 CaMgO_coord_3162
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3162


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3163 CaMgO_coord_3163
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3163


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3164 CaMgO_coord_3164
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3164


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3165 CaMgO_coord_3165
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3165


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3166 CaMgO_coord_3166
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3166


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3167 CaMgO_coord_3167
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3167


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3168 CaMgO_coord_3168
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3168


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3169 CaMgO_coord_3169
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3169


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3170 CaMgO_coord_3170
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3170


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3171 CaMgO_coord_3171
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3171


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3172 CaMgO_coord_3172
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3172


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3173 CaMgO_coord_3173
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3173


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3174 CaMgO_coord_3174
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3174


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3175 CaMgO_coord_3175
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3175


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3176 CaMgO_coord_3176
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3176


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3177 CaMgO_coord_3177
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3177


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3178 CaMgO_coord_3178
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3178


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3179 CaMgO_coord_3179
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3179


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3180 CaMgO_coord_3180
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3180


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3181 CaMgO_coord_3181
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3181


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3182 CaMgO_coord_3182
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3182


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3183 CaMgO_coord_3183
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3183


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3184 CaMgO_coord_3184
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3184


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3185 CaMgO_coord_3185
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3185


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3186 CaMgO_coord_3186
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3186


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3187 CaMgO_coord_3187
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3187


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3188 CaMgO_coord_3188
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3188


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3189 CaMgO_coord_3189
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3189


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3190 CaMgO_coord_3190
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3190


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3191 CaMgO_coord_3191
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3191


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3192 CaMgO_coord_3192
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3192


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3193 CaMgO_coord_3193
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3193


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3194 CaMgO_coord_3194
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3194


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3195 CaMgO_coord_3195
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3195


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3196 CaMgO_coord_3196
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3196


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3197 CaMgO_coord_3197
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3197


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3198 CaMgO_coord_3198
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3198


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3199 CaMgO_coord_3199
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3199


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3200 CaMgO_coord_3200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3200


