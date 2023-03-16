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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3201 CaMgO_coord_3201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3202 CaMgO_coord_3202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3203 CaMgO_coord_3203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3204 CaMgO_coord_3204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3205 CaMgO_coord_3205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3206 CaMgO_coord_3206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3207 CaMgO_coord_3207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3208 CaMgO_coord_3208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3209 CaMgO_coord_3209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3210 CaMgO_coord_3210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3211 CaMgO_coord_3211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3212 CaMgO_coord_3212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3213 CaMgO_coord_3213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3214 CaMgO_coord_3214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3215 CaMgO_coord_3215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3216 CaMgO_coord_3216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3217 CaMgO_coord_3217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3218 CaMgO_coord_3218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3219 CaMgO_coord_3219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3220 CaMgO_coord_3220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3221 CaMgO_coord_3221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3222 CaMgO_coord_3222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3223 CaMgO_coord_3223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3224 CaMgO_coord_3224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3225 CaMgO_coord_3225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3226 CaMgO_coord_3226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3227 CaMgO_coord_3227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3228 CaMgO_coord_3228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3229 CaMgO_coord_3229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3230 CaMgO_coord_3230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3231 CaMgO_coord_3231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3232 CaMgO_coord_3232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3233 CaMgO_coord_3233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3234 CaMgO_coord_3234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3235 CaMgO_coord_3235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3236 CaMgO_coord_3236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3237 CaMgO_coord_3237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3238 CaMgO_coord_3238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3239 CaMgO_coord_3239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3240 CaMgO_coord_3240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3241 CaMgO_coord_3241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3242 CaMgO_coord_3242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3243 CaMgO_coord_3243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3244 CaMgO_coord_3244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3245 CaMgO_coord_3245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3246 CaMgO_coord_3246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3247 CaMgO_coord_3247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3248 CaMgO_coord_3248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3249 CaMgO_coord_3249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3250 CaMgO_coord_3250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3251 CaMgO_coord_3251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3252 CaMgO_coord_3252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3253 CaMgO_coord_3253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3254 CaMgO_coord_3254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3255 CaMgO_coord_3255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3256 CaMgO_coord_3256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3257 CaMgO_coord_3257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3258 CaMgO_coord_3258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3259 CaMgO_coord_3259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3260 CaMgO_coord_3260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3261 CaMgO_coord_3261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3262 CaMgO_coord_3262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3263 CaMgO_coord_3263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3264 CaMgO_coord_3264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3265 CaMgO_coord_3265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3266 CaMgO_coord_3266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3267 CaMgO_coord_3267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3268 CaMgO_coord_3268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3269 CaMgO_coord_3269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3270 CaMgO_coord_3270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3271 CaMgO_coord_3271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3272 CaMgO_coord_3272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3273 CaMgO_coord_3273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3274 CaMgO_coord_3274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3275 CaMgO_coord_3275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3276 CaMgO_coord_3276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3277 CaMgO_coord_3277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3278 CaMgO_coord_3278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3279 CaMgO_coord_3279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3280 CaMgO_coord_3280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3281 CaMgO_coord_3281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3282 CaMgO_coord_3282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3283 CaMgO_coord_3283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3284 CaMgO_coord_3284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3285 CaMgO_coord_3285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3286 CaMgO_coord_3286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3287 CaMgO_coord_3287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3288 CaMgO_coord_3288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3289 CaMgO_coord_3289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3290 CaMgO_coord_3290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3291 CaMgO_coord_3291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3292 CaMgO_coord_3292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3293 CaMgO_coord_3293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3294 CaMgO_coord_3294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3295 CaMgO_coord_3295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3296 CaMgO_coord_3296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3297 CaMgO_coord_3297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3298 CaMgO_coord_3298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3299 CaMgO_coord_3299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3299


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3300 CaMgO_coord_3300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3300


