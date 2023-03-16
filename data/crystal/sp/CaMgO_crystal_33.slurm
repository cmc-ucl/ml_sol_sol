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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3299


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3300


