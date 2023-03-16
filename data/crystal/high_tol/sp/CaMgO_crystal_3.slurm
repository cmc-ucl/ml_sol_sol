#!/bin/bash
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-gc-smw    
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_200


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_299


