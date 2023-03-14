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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_201 CaMgO_coord_201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_202 CaMgO_coord_202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_203 CaMgO_coord_203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_204 CaMgO_coord_204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_205 CaMgO_coord_205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_206 CaMgO_coord_206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_207 CaMgO_coord_207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_208 CaMgO_coord_208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_209 CaMgO_coord_209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_210 CaMgO_coord_210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_211 CaMgO_coord_211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_212 CaMgO_coord_212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_213 CaMgO_coord_213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_214 CaMgO_coord_214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_215 CaMgO_coord_215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_216 CaMgO_coord_216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_217 CaMgO_coord_217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_218 CaMgO_coord_218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_219 CaMgO_coord_219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_220 CaMgO_coord_220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_221 CaMgO_coord_221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_222 CaMgO_coord_222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_223 CaMgO_coord_223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_224 CaMgO_coord_224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_225 CaMgO_coord_225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_226 CaMgO_coord_226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_227 CaMgO_coord_227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_228 CaMgO_coord_228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_229 CaMgO_coord_229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_230 CaMgO_coord_230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_231 CaMgO_coord_231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_232 CaMgO_coord_232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_233 CaMgO_coord_233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_234 CaMgO_coord_234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_235 CaMgO_coord_235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_236 CaMgO_coord_236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_237 CaMgO_coord_237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_238 CaMgO_coord_238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_239 CaMgO_coord_239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_240 CaMgO_coord_240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_241 CaMgO_coord_241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_242 CaMgO_coord_242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_243 CaMgO_coord_243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_244 CaMgO_coord_244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_245 CaMgO_coord_245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_246 CaMgO_coord_246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_247 CaMgO_coord_247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_248 CaMgO_coord_248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_249 CaMgO_coord_249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_250 CaMgO_coord_250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_251 CaMgO_coord_251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_252 CaMgO_coord_252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_253 CaMgO_coord_253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_254 CaMgO_coord_254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_255 CaMgO_coord_255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_256 CaMgO_coord_256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_257 CaMgO_coord_257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_258 CaMgO_coord_258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_259 CaMgO_coord_259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_260 CaMgO_coord_260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_261 CaMgO_coord_261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_262 CaMgO_coord_262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_263 CaMgO_coord_263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_264 CaMgO_coord_264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_265 CaMgO_coord_265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_266 CaMgO_coord_266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_267 CaMgO_coord_267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_268 CaMgO_coord_268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_269 CaMgO_coord_269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_270 CaMgO_coord_270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_271 CaMgO_coord_271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_272 CaMgO_coord_272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_273 CaMgO_coord_273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_274 CaMgO_coord_274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_275 CaMgO_coord_275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_276 CaMgO_coord_276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_277 CaMgO_coord_277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_278 CaMgO_coord_278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_279 CaMgO_coord_279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_280 CaMgO_coord_280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_281 CaMgO_coord_281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_282 CaMgO_coord_282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_283 CaMgO_coord_283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_284 CaMgO_coord_284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_285 CaMgO_coord_285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_286 CaMgO_coord_286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_287 CaMgO_coord_287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_288 CaMgO_coord_288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_289 CaMgO_coord_289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_290 CaMgO_coord_290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_291 CaMgO_coord_291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_292 CaMgO_coord_292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_293 CaMgO_coord_293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_294 CaMgO_coord_294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_295 CaMgO_coord_295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_296 CaMgO_coord_296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_297 CaMgO_coord_297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_298 CaMgO_coord_298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_299 CaMgO_coord_299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_299


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_300 CaMgO_coord_300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_300


