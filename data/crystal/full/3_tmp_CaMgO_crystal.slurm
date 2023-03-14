#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --time=12:00:00
#SBATCH --nodes=100
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=e05-power-smw
#SBATCH --partition=standard
#SBATCH --qos=standard   

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0
export OMP_NUM_THREADS=1

# Run calculations
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_200 CaMgO_full_tmp_200 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_201 CaMgO_full_tmp_201 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_202 CaMgO_full_tmp_202 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_203 CaMgO_full_tmp_203 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_204 CaMgO_full_tmp_204 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_205 CaMgO_full_tmp_205 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_206 CaMgO_full_tmp_206 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_207 CaMgO_full_tmp_207 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_208 CaMgO_full_tmp_208 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_209 CaMgO_full_tmp_209 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_210 CaMgO_full_tmp_210 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_211 CaMgO_full_tmp_211 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_212 CaMgO_full_tmp_212 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_213 CaMgO_full_tmp_213 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_214 CaMgO_full_tmp_214 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_215 CaMgO_full_tmp_215 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_216 CaMgO_full_tmp_216 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_217 CaMgO_full_tmp_217 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_218 CaMgO_full_tmp_218 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_219 CaMgO_full_tmp_219 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_220 CaMgO_full_tmp_220 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_221 CaMgO_full_tmp_221 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_222 CaMgO_full_tmp_222 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_223 CaMgO_full_tmp_223 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_224 CaMgO_full_tmp_224 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_225 CaMgO_full_tmp_225 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_226 CaMgO_full_tmp_226 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_227 CaMgO_full_tmp_227 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_228 CaMgO_full_tmp_228 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_229 CaMgO_full_tmp_229 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_230 CaMgO_full_tmp_230 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_231 CaMgO_full_tmp_231 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_232 CaMgO_full_tmp_232 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_233 CaMgO_full_tmp_233 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_234 CaMgO_full_tmp_234 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_235 CaMgO_full_tmp_235 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_236 CaMgO_full_tmp_236 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_237 CaMgO_full_tmp_237 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_238 CaMgO_full_tmp_238 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_239 CaMgO_full_tmp_239 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_240 CaMgO_full_tmp_240 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_241 CaMgO_full_tmp_241 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_242 CaMgO_full_tmp_242 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_243 CaMgO_full_tmp_243 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_244 CaMgO_full_tmp_244 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_245 CaMgO_full_tmp_245 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_246 CaMgO_full_tmp_246 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_247 CaMgO_full_tmp_247 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_248 CaMgO_full_tmp_248 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_249 CaMgO_full_tmp_249 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_250 CaMgO_full_tmp_250 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_251 CaMgO_full_tmp_251 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_252 CaMgO_full_tmp_252 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_253 CaMgO_full_tmp_253 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_254 CaMgO_full_tmp_254 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_255 CaMgO_full_tmp_255 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_256 CaMgO_full_tmp_256 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_257 CaMgO_full_tmp_257 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_258 CaMgO_full_tmp_258 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_259 CaMgO_full_tmp_259 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_260 CaMgO_full_tmp_260 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_261 CaMgO_full_tmp_261 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_262 CaMgO_full_tmp_262 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_263 CaMgO_full_tmp_263 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_264 CaMgO_full_tmp_264 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_265 CaMgO_full_tmp_265 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_266 CaMgO_full_tmp_266 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_267 CaMgO_full_tmp_267 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_268 CaMgO_full_tmp_268 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_269 CaMgO_full_tmp_269 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_270 CaMgO_full_tmp_270 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_271 CaMgO_full_tmp_271 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_272 CaMgO_full_tmp_272 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_273 CaMgO_full_tmp_273 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_274 CaMgO_full_tmp_274 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_275 CaMgO_full_tmp_275 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_276 CaMgO_full_tmp_276 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_277 CaMgO_full_tmp_277 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_278 CaMgO_full_tmp_278 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_279 CaMgO_full_tmp_279 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_280 CaMgO_full_tmp_280 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_281 CaMgO_full_tmp_281 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_282 CaMgO_full_tmp_282 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_283 CaMgO_full_tmp_283 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_284 CaMgO_full_tmp_284 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_285 CaMgO_full_tmp_285 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_286 CaMgO_full_tmp_286 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_287 CaMgO_full_tmp_287 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_288 CaMgO_full_tmp_288 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_289 CaMgO_full_tmp_289 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_290 CaMgO_full_tmp_290 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_291 CaMgO_full_tmp_291 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_292 CaMgO_full_tmp_292 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_293 CaMgO_full_tmp_293 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_294 CaMgO_full_tmp_294 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_295 CaMgO_full_tmp_295 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_296 CaMgO_full_tmp_296 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_297 CaMgO_full_tmp_297 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_298 CaMgO_full_tmp_298 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_299 CaMgO_full_tmp_299 &

wait
mv CaMgO_full_tmp_200.f9 CaMgO_full_200.f9
mv CaMgO_full_tmp_200.gui CaMgO_full_200.gui
mv CaMgO_full_tmp_201.f9 CaMgO_full_201.f9
mv CaMgO_full_tmp_201.gui CaMgO_full_201.gui
mv CaMgO_full_tmp_202.f9 CaMgO_full_202.f9
mv CaMgO_full_tmp_202.gui CaMgO_full_202.gui
mv CaMgO_full_tmp_203.f9 CaMgO_full_203.f9
mv CaMgO_full_tmp_203.gui CaMgO_full_203.gui
mv CaMgO_full_tmp_204.f9 CaMgO_full_204.f9
mv CaMgO_full_tmp_204.gui CaMgO_full_204.gui
mv CaMgO_full_tmp_205.f9 CaMgO_full_205.f9
mv CaMgO_full_tmp_205.gui CaMgO_full_205.gui
mv CaMgO_full_tmp_206.f9 CaMgO_full_206.f9
mv CaMgO_full_tmp_206.gui CaMgO_full_206.gui
mv CaMgO_full_tmp_207.f9 CaMgO_full_207.f9
mv CaMgO_full_tmp_207.gui CaMgO_full_207.gui
mv CaMgO_full_tmp_208.f9 CaMgO_full_208.f9
mv CaMgO_full_tmp_208.gui CaMgO_full_208.gui
mv CaMgO_full_tmp_209.f9 CaMgO_full_209.f9
mv CaMgO_full_tmp_209.gui CaMgO_full_209.gui
mv CaMgO_full_tmp_210.f9 CaMgO_full_210.f9
mv CaMgO_full_tmp_210.gui CaMgO_full_210.gui
mv CaMgO_full_tmp_211.f9 CaMgO_full_211.f9
mv CaMgO_full_tmp_211.gui CaMgO_full_211.gui
mv CaMgO_full_tmp_212.f9 CaMgO_full_212.f9
mv CaMgO_full_tmp_212.gui CaMgO_full_212.gui
mv CaMgO_full_tmp_213.f9 CaMgO_full_213.f9
mv CaMgO_full_tmp_213.gui CaMgO_full_213.gui
mv CaMgO_full_tmp_214.f9 CaMgO_full_214.f9
mv CaMgO_full_tmp_214.gui CaMgO_full_214.gui
mv CaMgO_full_tmp_215.f9 CaMgO_full_215.f9
mv CaMgO_full_tmp_215.gui CaMgO_full_215.gui
mv CaMgO_full_tmp_216.f9 CaMgO_full_216.f9
mv CaMgO_full_tmp_216.gui CaMgO_full_216.gui
mv CaMgO_full_tmp_217.f9 CaMgO_full_217.f9
mv CaMgO_full_tmp_217.gui CaMgO_full_217.gui
mv CaMgO_full_tmp_218.f9 CaMgO_full_218.f9
mv CaMgO_full_tmp_218.gui CaMgO_full_218.gui
mv CaMgO_full_tmp_219.f9 CaMgO_full_219.f9
mv CaMgO_full_tmp_219.gui CaMgO_full_219.gui
mv CaMgO_full_tmp_220.f9 CaMgO_full_220.f9
mv CaMgO_full_tmp_220.gui CaMgO_full_220.gui
mv CaMgO_full_tmp_221.f9 CaMgO_full_221.f9
mv CaMgO_full_tmp_221.gui CaMgO_full_221.gui
mv CaMgO_full_tmp_222.f9 CaMgO_full_222.f9
mv CaMgO_full_tmp_222.gui CaMgO_full_222.gui
mv CaMgO_full_tmp_223.f9 CaMgO_full_223.f9
mv CaMgO_full_tmp_223.gui CaMgO_full_223.gui
mv CaMgO_full_tmp_224.f9 CaMgO_full_224.f9
mv CaMgO_full_tmp_224.gui CaMgO_full_224.gui
mv CaMgO_full_tmp_225.f9 CaMgO_full_225.f9
mv CaMgO_full_tmp_225.gui CaMgO_full_225.gui
mv CaMgO_full_tmp_226.f9 CaMgO_full_226.f9
mv CaMgO_full_tmp_226.gui CaMgO_full_226.gui
mv CaMgO_full_tmp_227.f9 CaMgO_full_227.f9
mv CaMgO_full_tmp_227.gui CaMgO_full_227.gui
mv CaMgO_full_tmp_228.f9 CaMgO_full_228.f9
mv CaMgO_full_tmp_228.gui CaMgO_full_228.gui
mv CaMgO_full_tmp_229.f9 CaMgO_full_229.f9
mv CaMgO_full_tmp_229.gui CaMgO_full_229.gui
mv CaMgO_full_tmp_230.f9 CaMgO_full_230.f9
mv CaMgO_full_tmp_230.gui CaMgO_full_230.gui
mv CaMgO_full_tmp_231.f9 CaMgO_full_231.f9
mv CaMgO_full_tmp_231.gui CaMgO_full_231.gui
mv CaMgO_full_tmp_232.f9 CaMgO_full_232.f9
mv CaMgO_full_tmp_232.gui CaMgO_full_232.gui
mv CaMgO_full_tmp_233.f9 CaMgO_full_233.f9
mv CaMgO_full_tmp_233.gui CaMgO_full_233.gui
mv CaMgO_full_tmp_234.f9 CaMgO_full_234.f9
mv CaMgO_full_tmp_234.gui CaMgO_full_234.gui
mv CaMgO_full_tmp_235.f9 CaMgO_full_235.f9
mv CaMgO_full_tmp_235.gui CaMgO_full_235.gui
mv CaMgO_full_tmp_236.f9 CaMgO_full_236.f9
mv CaMgO_full_tmp_236.gui CaMgO_full_236.gui
mv CaMgO_full_tmp_237.f9 CaMgO_full_237.f9
mv CaMgO_full_tmp_237.gui CaMgO_full_237.gui
mv CaMgO_full_tmp_238.f9 CaMgO_full_238.f9
mv CaMgO_full_tmp_238.gui CaMgO_full_238.gui
mv CaMgO_full_tmp_239.f9 CaMgO_full_239.f9
mv CaMgO_full_tmp_239.gui CaMgO_full_239.gui
mv CaMgO_full_tmp_240.f9 CaMgO_full_240.f9
mv CaMgO_full_tmp_240.gui CaMgO_full_240.gui
mv CaMgO_full_tmp_241.f9 CaMgO_full_241.f9
mv CaMgO_full_tmp_241.gui CaMgO_full_241.gui
mv CaMgO_full_tmp_242.f9 CaMgO_full_242.f9
mv CaMgO_full_tmp_242.gui CaMgO_full_242.gui
mv CaMgO_full_tmp_243.f9 CaMgO_full_243.f9
mv CaMgO_full_tmp_243.gui CaMgO_full_243.gui
mv CaMgO_full_tmp_244.f9 CaMgO_full_244.f9
mv CaMgO_full_tmp_244.gui CaMgO_full_244.gui
mv CaMgO_full_tmp_245.f9 CaMgO_full_245.f9
mv CaMgO_full_tmp_245.gui CaMgO_full_245.gui
mv CaMgO_full_tmp_246.f9 CaMgO_full_246.f9
mv CaMgO_full_tmp_246.gui CaMgO_full_246.gui
mv CaMgO_full_tmp_247.f9 CaMgO_full_247.f9
mv CaMgO_full_tmp_247.gui CaMgO_full_247.gui
mv CaMgO_full_tmp_248.f9 CaMgO_full_248.f9
mv CaMgO_full_tmp_248.gui CaMgO_full_248.gui
mv CaMgO_full_tmp_249.f9 CaMgO_full_249.f9
mv CaMgO_full_tmp_249.gui CaMgO_full_249.gui
mv CaMgO_full_tmp_250.f9 CaMgO_full_250.f9
mv CaMgO_full_tmp_250.gui CaMgO_full_250.gui
mv CaMgO_full_tmp_251.f9 CaMgO_full_251.f9
mv CaMgO_full_tmp_251.gui CaMgO_full_251.gui
mv CaMgO_full_tmp_252.f9 CaMgO_full_252.f9
mv CaMgO_full_tmp_252.gui CaMgO_full_252.gui
mv CaMgO_full_tmp_253.f9 CaMgO_full_253.f9
mv CaMgO_full_tmp_253.gui CaMgO_full_253.gui
mv CaMgO_full_tmp_254.f9 CaMgO_full_254.f9
mv CaMgO_full_tmp_254.gui CaMgO_full_254.gui
mv CaMgO_full_tmp_255.f9 CaMgO_full_255.f9
mv CaMgO_full_tmp_255.gui CaMgO_full_255.gui
mv CaMgO_full_tmp_256.f9 CaMgO_full_256.f9
mv CaMgO_full_tmp_256.gui CaMgO_full_256.gui
mv CaMgO_full_tmp_257.f9 CaMgO_full_257.f9
mv CaMgO_full_tmp_257.gui CaMgO_full_257.gui
mv CaMgO_full_tmp_258.f9 CaMgO_full_258.f9
mv CaMgO_full_tmp_258.gui CaMgO_full_258.gui
mv CaMgO_full_tmp_259.f9 CaMgO_full_259.f9
mv CaMgO_full_tmp_259.gui CaMgO_full_259.gui
mv CaMgO_full_tmp_260.f9 CaMgO_full_260.f9
mv CaMgO_full_tmp_260.gui CaMgO_full_260.gui
mv CaMgO_full_tmp_261.f9 CaMgO_full_261.f9
mv CaMgO_full_tmp_261.gui CaMgO_full_261.gui
mv CaMgO_full_tmp_262.f9 CaMgO_full_262.f9
mv CaMgO_full_tmp_262.gui CaMgO_full_262.gui
mv CaMgO_full_tmp_263.f9 CaMgO_full_263.f9
mv CaMgO_full_tmp_263.gui CaMgO_full_263.gui
mv CaMgO_full_tmp_264.f9 CaMgO_full_264.f9
mv CaMgO_full_tmp_264.gui CaMgO_full_264.gui
mv CaMgO_full_tmp_265.f9 CaMgO_full_265.f9
mv CaMgO_full_tmp_265.gui CaMgO_full_265.gui
mv CaMgO_full_tmp_266.f9 CaMgO_full_266.f9
mv CaMgO_full_tmp_266.gui CaMgO_full_266.gui
mv CaMgO_full_tmp_267.f9 CaMgO_full_267.f9
mv CaMgO_full_tmp_267.gui CaMgO_full_267.gui
mv CaMgO_full_tmp_268.f9 CaMgO_full_268.f9
mv CaMgO_full_tmp_268.gui CaMgO_full_268.gui
mv CaMgO_full_tmp_269.f9 CaMgO_full_269.f9
mv CaMgO_full_tmp_269.gui CaMgO_full_269.gui
mv CaMgO_full_tmp_270.f9 CaMgO_full_270.f9
mv CaMgO_full_tmp_270.gui CaMgO_full_270.gui
mv CaMgO_full_tmp_271.f9 CaMgO_full_271.f9
mv CaMgO_full_tmp_271.gui CaMgO_full_271.gui
mv CaMgO_full_tmp_272.f9 CaMgO_full_272.f9
mv CaMgO_full_tmp_272.gui CaMgO_full_272.gui
mv CaMgO_full_tmp_273.f9 CaMgO_full_273.f9
mv CaMgO_full_tmp_273.gui CaMgO_full_273.gui
mv CaMgO_full_tmp_274.f9 CaMgO_full_274.f9
mv CaMgO_full_tmp_274.gui CaMgO_full_274.gui
mv CaMgO_full_tmp_275.f9 CaMgO_full_275.f9
mv CaMgO_full_tmp_275.gui CaMgO_full_275.gui
mv CaMgO_full_tmp_276.f9 CaMgO_full_276.f9
mv CaMgO_full_tmp_276.gui CaMgO_full_276.gui
mv CaMgO_full_tmp_277.f9 CaMgO_full_277.f9
mv CaMgO_full_tmp_277.gui CaMgO_full_277.gui
mv CaMgO_full_tmp_278.f9 CaMgO_full_278.f9
mv CaMgO_full_tmp_278.gui CaMgO_full_278.gui
mv CaMgO_full_tmp_279.f9 CaMgO_full_279.f9
mv CaMgO_full_tmp_279.gui CaMgO_full_279.gui
mv CaMgO_full_tmp_280.f9 CaMgO_full_280.f9
mv CaMgO_full_tmp_280.gui CaMgO_full_280.gui
mv CaMgO_full_tmp_281.f9 CaMgO_full_281.f9
mv CaMgO_full_tmp_281.gui CaMgO_full_281.gui
mv CaMgO_full_tmp_282.f9 CaMgO_full_282.f9
mv CaMgO_full_tmp_282.gui CaMgO_full_282.gui
mv CaMgO_full_tmp_283.f9 CaMgO_full_283.f9
mv CaMgO_full_tmp_283.gui CaMgO_full_283.gui
mv CaMgO_full_tmp_284.f9 CaMgO_full_284.f9
mv CaMgO_full_tmp_284.gui CaMgO_full_284.gui
mv CaMgO_full_tmp_285.f9 CaMgO_full_285.f9
mv CaMgO_full_tmp_285.gui CaMgO_full_285.gui
mv CaMgO_full_tmp_286.f9 CaMgO_full_286.f9
mv CaMgO_full_tmp_286.gui CaMgO_full_286.gui
mv CaMgO_full_tmp_287.f9 CaMgO_full_287.f9
mv CaMgO_full_tmp_287.gui CaMgO_full_287.gui
mv CaMgO_full_tmp_288.f9 CaMgO_full_288.f9
mv CaMgO_full_tmp_288.gui CaMgO_full_288.gui
mv CaMgO_full_tmp_289.f9 CaMgO_full_289.f9
mv CaMgO_full_tmp_289.gui CaMgO_full_289.gui
mv CaMgO_full_tmp_290.f9 CaMgO_full_290.f9
mv CaMgO_full_tmp_290.gui CaMgO_full_290.gui
mv CaMgO_full_tmp_291.f9 CaMgO_full_291.f9
mv CaMgO_full_tmp_291.gui CaMgO_full_291.gui
mv CaMgO_full_tmp_292.f9 CaMgO_full_292.f9
mv CaMgO_full_tmp_292.gui CaMgO_full_292.gui
mv CaMgO_full_tmp_293.f9 CaMgO_full_293.f9
mv CaMgO_full_tmp_293.gui CaMgO_full_293.gui
mv CaMgO_full_tmp_294.f9 CaMgO_full_294.f9
mv CaMgO_full_tmp_294.gui CaMgO_full_294.gui
mv CaMgO_full_tmp_295.f9 CaMgO_full_295.f9
mv CaMgO_full_tmp_295.gui CaMgO_full_295.gui
mv CaMgO_full_tmp_296.f9 CaMgO_full_296.f9
mv CaMgO_full_tmp_296.gui CaMgO_full_296.gui
mv CaMgO_full_tmp_297.f9 CaMgO_full_297.f9
mv CaMgO_full_tmp_297.gui CaMgO_full_297.gui
mv CaMgO_full_tmp_298.f9 CaMgO_full_298.f9
mv CaMgO_full_tmp_298.gui CaMgO_full_298.gui
mv CaMgO_full_tmp_299.f9 CaMgO_full_299.f9
mv CaMgO_full_tmp_299.gui CaMgO_full_299.gui
