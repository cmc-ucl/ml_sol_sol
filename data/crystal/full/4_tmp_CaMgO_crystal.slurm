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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_300 CaMgO_full_tmp_300 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_301 CaMgO_full_tmp_301 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_302 CaMgO_full_tmp_302 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_303 CaMgO_full_tmp_303 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_304 CaMgO_full_tmp_304 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_305 CaMgO_full_tmp_305 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_306 CaMgO_full_tmp_306 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_307 CaMgO_full_tmp_307 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_308 CaMgO_full_tmp_308 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_309 CaMgO_full_tmp_309 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_310 CaMgO_full_tmp_310 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_311 CaMgO_full_tmp_311 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_312 CaMgO_full_tmp_312 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_313 CaMgO_full_tmp_313 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_314 CaMgO_full_tmp_314 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_315 CaMgO_full_tmp_315 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_316 CaMgO_full_tmp_316 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_317 CaMgO_full_tmp_317 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_318 CaMgO_full_tmp_318 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_319 CaMgO_full_tmp_319 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_320 CaMgO_full_tmp_320 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_321 CaMgO_full_tmp_321 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_322 CaMgO_full_tmp_322 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_323 CaMgO_full_tmp_323 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_324 CaMgO_full_tmp_324 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_325 CaMgO_full_tmp_325 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_326 CaMgO_full_tmp_326 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_327 CaMgO_full_tmp_327 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_328 CaMgO_full_tmp_328 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_329 CaMgO_full_tmp_329 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_330 CaMgO_full_tmp_330 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_331 CaMgO_full_tmp_331 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_332 CaMgO_full_tmp_332 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_333 CaMgO_full_tmp_333 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_334 CaMgO_full_tmp_334 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_335 CaMgO_full_tmp_335 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_336 CaMgO_full_tmp_336 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_337 CaMgO_full_tmp_337 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_338 CaMgO_full_tmp_338 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_339 CaMgO_full_tmp_339 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_340 CaMgO_full_tmp_340 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_341 CaMgO_full_tmp_341 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_342 CaMgO_full_tmp_342 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_343 CaMgO_full_tmp_343 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_344 CaMgO_full_tmp_344 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_345 CaMgO_full_tmp_345 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_346 CaMgO_full_tmp_346 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_347 CaMgO_full_tmp_347 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_348 CaMgO_full_tmp_348 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_349 CaMgO_full_tmp_349 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_350 CaMgO_full_tmp_350 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_351 CaMgO_full_tmp_351 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_352 CaMgO_full_tmp_352 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_353 CaMgO_full_tmp_353 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_354 CaMgO_full_tmp_354 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_355 CaMgO_full_tmp_355 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_356 CaMgO_full_tmp_356 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_357 CaMgO_full_tmp_357 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_358 CaMgO_full_tmp_358 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_359 CaMgO_full_tmp_359 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_360 CaMgO_full_tmp_360 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_361 CaMgO_full_tmp_361 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_362 CaMgO_full_tmp_362 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_363 CaMgO_full_tmp_363 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_364 CaMgO_full_tmp_364 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_365 CaMgO_full_tmp_365 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_366 CaMgO_full_tmp_366 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_367 CaMgO_full_tmp_367 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_368 CaMgO_full_tmp_368 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_369 CaMgO_full_tmp_369 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_370 CaMgO_full_tmp_370 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_371 CaMgO_full_tmp_371 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_372 CaMgO_full_tmp_372 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_373 CaMgO_full_tmp_373 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_374 CaMgO_full_tmp_374 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_375 CaMgO_full_tmp_375 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_376 CaMgO_full_tmp_376 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_377 CaMgO_full_tmp_377 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_378 CaMgO_full_tmp_378 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_379 CaMgO_full_tmp_379 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_380 CaMgO_full_tmp_380 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_381 CaMgO_full_tmp_381 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_382 CaMgO_full_tmp_382 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_383 CaMgO_full_tmp_383 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_384 CaMgO_full_tmp_384 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_385 CaMgO_full_tmp_385 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_386 CaMgO_full_tmp_386 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_387 CaMgO_full_tmp_387 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_388 CaMgO_full_tmp_388 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_389 CaMgO_full_tmp_389 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_390 CaMgO_full_tmp_390 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_391 CaMgO_full_tmp_391 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_392 CaMgO_full_tmp_392 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_393 CaMgO_full_tmp_393 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_394 CaMgO_full_tmp_394 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_395 CaMgO_full_tmp_395 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_396 CaMgO_full_tmp_396 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_397 CaMgO_full_tmp_397 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_398 CaMgO_full_tmp_398 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_399 CaMgO_full_tmp_399 &

wait
mv CaMgO_full_tmp_300.f9 CaMgO_full_300.f9
mv CaMgO_full_tmp_300.gui CaMgO_full_300.gui
mv CaMgO_full_tmp_301.f9 CaMgO_full_301.f9
mv CaMgO_full_tmp_301.gui CaMgO_full_301.gui
mv CaMgO_full_tmp_302.f9 CaMgO_full_302.f9
mv CaMgO_full_tmp_302.gui CaMgO_full_302.gui
mv CaMgO_full_tmp_303.f9 CaMgO_full_303.f9
mv CaMgO_full_tmp_303.gui CaMgO_full_303.gui
mv CaMgO_full_tmp_304.f9 CaMgO_full_304.f9
mv CaMgO_full_tmp_304.gui CaMgO_full_304.gui
mv CaMgO_full_tmp_305.f9 CaMgO_full_305.f9
mv CaMgO_full_tmp_305.gui CaMgO_full_305.gui
mv CaMgO_full_tmp_306.f9 CaMgO_full_306.f9
mv CaMgO_full_tmp_306.gui CaMgO_full_306.gui
mv CaMgO_full_tmp_307.f9 CaMgO_full_307.f9
mv CaMgO_full_tmp_307.gui CaMgO_full_307.gui
mv CaMgO_full_tmp_308.f9 CaMgO_full_308.f9
mv CaMgO_full_tmp_308.gui CaMgO_full_308.gui
mv CaMgO_full_tmp_309.f9 CaMgO_full_309.f9
mv CaMgO_full_tmp_309.gui CaMgO_full_309.gui
mv CaMgO_full_tmp_310.f9 CaMgO_full_310.f9
mv CaMgO_full_tmp_310.gui CaMgO_full_310.gui
mv CaMgO_full_tmp_311.f9 CaMgO_full_311.f9
mv CaMgO_full_tmp_311.gui CaMgO_full_311.gui
mv CaMgO_full_tmp_312.f9 CaMgO_full_312.f9
mv CaMgO_full_tmp_312.gui CaMgO_full_312.gui
mv CaMgO_full_tmp_313.f9 CaMgO_full_313.f9
mv CaMgO_full_tmp_313.gui CaMgO_full_313.gui
mv CaMgO_full_tmp_314.f9 CaMgO_full_314.f9
mv CaMgO_full_tmp_314.gui CaMgO_full_314.gui
mv CaMgO_full_tmp_315.f9 CaMgO_full_315.f9
mv CaMgO_full_tmp_315.gui CaMgO_full_315.gui
mv CaMgO_full_tmp_316.f9 CaMgO_full_316.f9
mv CaMgO_full_tmp_316.gui CaMgO_full_316.gui
mv CaMgO_full_tmp_317.f9 CaMgO_full_317.f9
mv CaMgO_full_tmp_317.gui CaMgO_full_317.gui
mv CaMgO_full_tmp_318.f9 CaMgO_full_318.f9
mv CaMgO_full_tmp_318.gui CaMgO_full_318.gui
mv CaMgO_full_tmp_319.f9 CaMgO_full_319.f9
mv CaMgO_full_tmp_319.gui CaMgO_full_319.gui
mv CaMgO_full_tmp_320.f9 CaMgO_full_320.f9
mv CaMgO_full_tmp_320.gui CaMgO_full_320.gui
mv CaMgO_full_tmp_321.f9 CaMgO_full_321.f9
mv CaMgO_full_tmp_321.gui CaMgO_full_321.gui
mv CaMgO_full_tmp_322.f9 CaMgO_full_322.f9
mv CaMgO_full_tmp_322.gui CaMgO_full_322.gui
mv CaMgO_full_tmp_323.f9 CaMgO_full_323.f9
mv CaMgO_full_tmp_323.gui CaMgO_full_323.gui
mv CaMgO_full_tmp_324.f9 CaMgO_full_324.f9
mv CaMgO_full_tmp_324.gui CaMgO_full_324.gui
mv CaMgO_full_tmp_325.f9 CaMgO_full_325.f9
mv CaMgO_full_tmp_325.gui CaMgO_full_325.gui
mv CaMgO_full_tmp_326.f9 CaMgO_full_326.f9
mv CaMgO_full_tmp_326.gui CaMgO_full_326.gui
mv CaMgO_full_tmp_327.f9 CaMgO_full_327.f9
mv CaMgO_full_tmp_327.gui CaMgO_full_327.gui
mv CaMgO_full_tmp_328.f9 CaMgO_full_328.f9
mv CaMgO_full_tmp_328.gui CaMgO_full_328.gui
mv CaMgO_full_tmp_329.f9 CaMgO_full_329.f9
mv CaMgO_full_tmp_329.gui CaMgO_full_329.gui
mv CaMgO_full_tmp_330.f9 CaMgO_full_330.f9
mv CaMgO_full_tmp_330.gui CaMgO_full_330.gui
mv CaMgO_full_tmp_331.f9 CaMgO_full_331.f9
mv CaMgO_full_tmp_331.gui CaMgO_full_331.gui
mv CaMgO_full_tmp_332.f9 CaMgO_full_332.f9
mv CaMgO_full_tmp_332.gui CaMgO_full_332.gui
mv CaMgO_full_tmp_333.f9 CaMgO_full_333.f9
mv CaMgO_full_tmp_333.gui CaMgO_full_333.gui
mv CaMgO_full_tmp_334.f9 CaMgO_full_334.f9
mv CaMgO_full_tmp_334.gui CaMgO_full_334.gui
mv CaMgO_full_tmp_335.f9 CaMgO_full_335.f9
mv CaMgO_full_tmp_335.gui CaMgO_full_335.gui
mv CaMgO_full_tmp_336.f9 CaMgO_full_336.f9
mv CaMgO_full_tmp_336.gui CaMgO_full_336.gui
mv CaMgO_full_tmp_337.f9 CaMgO_full_337.f9
mv CaMgO_full_tmp_337.gui CaMgO_full_337.gui
mv CaMgO_full_tmp_338.f9 CaMgO_full_338.f9
mv CaMgO_full_tmp_338.gui CaMgO_full_338.gui
mv CaMgO_full_tmp_339.f9 CaMgO_full_339.f9
mv CaMgO_full_tmp_339.gui CaMgO_full_339.gui
mv CaMgO_full_tmp_340.f9 CaMgO_full_340.f9
mv CaMgO_full_tmp_340.gui CaMgO_full_340.gui
mv CaMgO_full_tmp_341.f9 CaMgO_full_341.f9
mv CaMgO_full_tmp_341.gui CaMgO_full_341.gui
mv CaMgO_full_tmp_342.f9 CaMgO_full_342.f9
mv CaMgO_full_tmp_342.gui CaMgO_full_342.gui
mv CaMgO_full_tmp_343.f9 CaMgO_full_343.f9
mv CaMgO_full_tmp_343.gui CaMgO_full_343.gui
mv CaMgO_full_tmp_344.f9 CaMgO_full_344.f9
mv CaMgO_full_tmp_344.gui CaMgO_full_344.gui
mv CaMgO_full_tmp_345.f9 CaMgO_full_345.f9
mv CaMgO_full_tmp_345.gui CaMgO_full_345.gui
mv CaMgO_full_tmp_346.f9 CaMgO_full_346.f9
mv CaMgO_full_tmp_346.gui CaMgO_full_346.gui
mv CaMgO_full_tmp_347.f9 CaMgO_full_347.f9
mv CaMgO_full_tmp_347.gui CaMgO_full_347.gui
mv CaMgO_full_tmp_348.f9 CaMgO_full_348.f9
mv CaMgO_full_tmp_348.gui CaMgO_full_348.gui
mv CaMgO_full_tmp_349.f9 CaMgO_full_349.f9
mv CaMgO_full_tmp_349.gui CaMgO_full_349.gui
mv CaMgO_full_tmp_350.f9 CaMgO_full_350.f9
mv CaMgO_full_tmp_350.gui CaMgO_full_350.gui
mv CaMgO_full_tmp_351.f9 CaMgO_full_351.f9
mv CaMgO_full_tmp_351.gui CaMgO_full_351.gui
mv CaMgO_full_tmp_352.f9 CaMgO_full_352.f9
mv CaMgO_full_tmp_352.gui CaMgO_full_352.gui
mv CaMgO_full_tmp_353.f9 CaMgO_full_353.f9
mv CaMgO_full_tmp_353.gui CaMgO_full_353.gui
mv CaMgO_full_tmp_354.f9 CaMgO_full_354.f9
mv CaMgO_full_tmp_354.gui CaMgO_full_354.gui
mv CaMgO_full_tmp_355.f9 CaMgO_full_355.f9
mv CaMgO_full_tmp_355.gui CaMgO_full_355.gui
mv CaMgO_full_tmp_356.f9 CaMgO_full_356.f9
mv CaMgO_full_tmp_356.gui CaMgO_full_356.gui
mv CaMgO_full_tmp_357.f9 CaMgO_full_357.f9
mv CaMgO_full_tmp_357.gui CaMgO_full_357.gui
mv CaMgO_full_tmp_358.f9 CaMgO_full_358.f9
mv CaMgO_full_tmp_358.gui CaMgO_full_358.gui
mv CaMgO_full_tmp_359.f9 CaMgO_full_359.f9
mv CaMgO_full_tmp_359.gui CaMgO_full_359.gui
mv CaMgO_full_tmp_360.f9 CaMgO_full_360.f9
mv CaMgO_full_tmp_360.gui CaMgO_full_360.gui
mv CaMgO_full_tmp_361.f9 CaMgO_full_361.f9
mv CaMgO_full_tmp_361.gui CaMgO_full_361.gui
mv CaMgO_full_tmp_362.f9 CaMgO_full_362.f9
mv CaMgO_full_tmp_362.gui CaMgO_full_362.gui
mv CaMgO_full_tmp_363.f9 CaMgO_full_363.f9
mv CaMgO_full_tmp_363.gui CaMgO_full_363.gui
mv CaMgO_full_tmp_364.f9 CaMgO_full_364.f9
mv CaMgO_full_tmp_364.gui CaMgO_full_364.gui
mv CaMgO_full_tmp_365.f9 CaMgO_full_365.f9
mv CaMgO_full_tmp_365.gui CaMgO_full_365.gui
mv CaMgO_full_tmp_366.f9 CaMgO_full_366.f9
mv CaMgO_full_tmp_366.gui CaMgO_full_366.gui
mv CaMgO_full_tmp_367.f9 CaMgO_full_367.f9
mv CaMgO_full_tmp_367.gui CaMgO_full_367.gui
mv CaMgO_full_tmp_368.f9 CaMgO_full_368.f9
mv CaMgO_full_tmp_368.gui CaMgO_full_368.gui
mv CaMgO_full_tmp_369.f9 CaMgO_full_369.f9
mv CaMgO_full_tmp_369.gui CaMgO_full_369.gui
mv CaMgO_full_tmp_370.f9 CaMgO_full_370.f9
mv CaMgO_full_tmp_370.gui CaMgO_full_370.gui
mv CaMgO_full_tmp_371.f9 CaMgO_full_371.f9
mv CaMgO_full_tmp_371.gui CaMgO_full_371.gui
mv CaMgO_full_tmp_372.f9 CaMgO_full_372.f9
mv CaMgO_full_tmp_372.gui CaMgO_full_372.gui
mv CaMgO_full_tmp_373.f9 CaMgO_full_373.f9
mv CaMgO_full_tmp_373.gui CaMgO_full_373.gui
mv CaMgO_full_tmp_374.f9 CaMgO_full_374.f9
mv CaMgO_full_tmp_374.gui CaMgO_full_374.gui
mv CaMgO_full_tmp_375.f9 CaMgO_full_375.f9
mv CaMgO_full_tmp_375.gui CaMgO_full_375.gui
mv CaMgO_full_tmp_376.f9 CaMgO_full_376.f9
mv CaMgO_full_tmp_376.gui CaMgO_full_376.gui
mv CaMgO_full_tmp_377.f9 CaMgO_full_377.f9
mv CaMgO_full_tmp_377.gui CaMgO_full_377.gui
mv CaMgO_full_tmp_378.f9 CaMgO_full_378.f9
mv CaMgO_full_tmp_378.gui CaMgO_full_378.gui
mv CaMgO_full_tmp_379.f9 CaMgO_full_379.f9
mv CaMgO_full_tmp_379.gui CaMgO_full_379.gui
mv CaMgO_full_tmp_380.f9 CaMgO_full_380.f9
mv CaMgO_full_tmp_380.gui CaMgO_full_380.gui
mv CaMgO_full_tmp_381.f9 CaMgO_full_381.f9
mv CaMgO_full_tmp_381.gui CaMgO_full_381.gui
mv CaMgO_full_tmp_382.f9 CaMgO_full_382.f9
mv CaMgO_full_tmp_382.gui CaMgO_full_382.gui
mv CaMgO_full_tmp_383.f9 CaMgO_full_383.f9
mv CaMgO_full_tmp_383.gui CaMgO_full_383.gui
mv CaMgO_full_tmp_384.f9 CaMgO_full_384.f9
mv CaMgO_full_tmp_384.gui CaMgO_full_384.gui
mv CaMgO_full_tmp_385.f9 CaMgO_full_385.f9
mv CaMgO_full_tmp_385.gui CaMgO_full_385.gui
mv CaMgO_full_tmp_386.f9 CaMgO_full_386.f9
mv CaMgO_full_tmp_386.gui CaMgO_full_386.gui
mv CaMgO_full_tmp_387.f9 CaMgO_full_387.f9
mv CaMgO_full_tmp_387.gui CaMgO_full_387.gui
mv CaMgO_full_tmp_388.f9 CaMgO_full_388.f9
mv CaMgO_full_tmp_388.gui CaMgO_full_388.gui
mv CaMgO_full_tmp_389.f9 CaMgO_full_389.f9
mv CaMgO_full_tmp_389.gui CaMgO_full_389.gui
mv CaMgO_full_tmp_390.f9 CaMgO_full_390.f9
mv CaMgO_full_tmp_390.gui CaMgO_full_390.gui
mv CaMgO_full_tmp_391.f9 CaMgO_full_391.f9
mv CaMgO_full_tmp_391.gui CaMgO_full_391.gui
mv CaMgO_full_tmp_392.f9 CaMgO_full_392.f9
mv CaMgO_full_tmp_392.gui CaMgO_full_392.gui
mv CaMgO_full_tmp_393.f9 CaMgO_full_393.f9
mv CaMgO_full_tmp_393.gui CaMgO_full_393.gui
mv CaMgO_full_tmp_394.f9 CaMgO_full_394.f9
mv CaMgO_full_tmp_394.gui CaMgO_full_394.gui
mv CaMgO_full_tmp_395.f9 CaMgO_full_395.f9
mv CaMgO_full_tmp_395.gui CaMgO_full_395.gui
mv CaMgO_full_tmp_396.f9 CaMgO_full_396.f9
mv CaMgO_full_tmp_396.gui CaMgO_full_396.gui
mv CaMgO_full_tmp_397.f9 CaMgO_full_397.f9
mv CaMgO_full_tmp_397.gui CaMgO_full_397.gui
mv CaMgO_full_tmp_398.f9 CaMgO_full_398.f9
mv CaMgO_full_tmp_398.gui CaMgO_full_398.gui
mv CaMgO_full_tmp_399.f9 CaMgO_full_399.f9
mv CaMgO_full_tmp_399.gui CaMgO_full_399.gui
