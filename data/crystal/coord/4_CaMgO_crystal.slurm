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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_301 CaMgO_coord_301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_302 CaMgO_coord_302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_303 CaMgO_coord_303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_304 CaMgO_coord_304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_305 CaMgO_coord_305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_306 CaMgO_coord_306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_307 CaMgO_coord_307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_308 CaMgO_coord_308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_309 CaMgO_coord_309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_310 CaMgO_coord_310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_311 CaMgO_coord_311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_312 CaMgO_coord_312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_313 CaMgO_coord_313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_314 CaMgO_coord_314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_315 CaMgO_coord_315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_316 CaMgO_coord_316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_317 CaMgO_coord_317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_318 CaMgO_coord_318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_319 CaMgO_coord_319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_320 CaMgO_coord_320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_321 CaMgO_coord_321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_322 CaMgO_coord_322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_323 CaMgO_coord_323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_324 CaMgO_coord_324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_325 CaMgO_coord_325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_326 CaMgO_coord_326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_327 CaMgO_coord_327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_328 CaMgO_coord_328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_329 CaMgO_coord_329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_330 CaMgO_coord_330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_331 CaMgO_coord_331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_332 CaMgO_coord_332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_333 CaMgO_coord_333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_334 CaMgO_coord_334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_335 CaMgO_coord_335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_336 CaMgO_coord_336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_337 CaMgO_coord_337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_338 CaMgO_coord_338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_339 CaMgO_coord_339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_340 CaMgO_coord_340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_341 CaMgO_coord_341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_342 CaMgO_coord_342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_343 CaMgO_coord_343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_344 CaMgO_coord_344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_345 CaMgO_coord_345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_346 CaMgO_coord_346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_347 CaMgO_coord_347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_348 CaMgO_coord_348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_349 CaMgO_coord_349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_350 CaMgO_coord_350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_351 CaMgO_coord_351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_352 CaMgO_coord_352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_353 CaMgO_coord_353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_354 CaMgO_coord_354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_355 CaMgO_coord_355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_356 CaMgO_coord_356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_357 CaMgO_coord_357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_358 CaMgO_coord_358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_359 CaMgO_coord_359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_360 CaMgO_coord_360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_361 CaMgO_coord_361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_362 CaMgO_coord_362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_363 CaMgO_coord_363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_364 CaMgO_coord_364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_365 CaMgO_coord_365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_366 CaMgO_coord_366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_367 CaMgO_coord_367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_368 CaMgO_coord_368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_369 CaMgO_coord_369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_370 CaMgO_coord_370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_371 CaMgO_coord_371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_372 CaMgO_coord_372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_373 CaMgO_coord_373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_374 CaMgO_coord_374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_375 CaMgO_coord_375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_376 CaMgO_coord_376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_377 CaMgO_coord_377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_378 CaMgO_coord_378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_379 CaMgO_coord_379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_380 CaMgO_coord_380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_381 CaMgO_coord_381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_382 CaMgO_coord_382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_383 CaMgO_coord_383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_384 CaMgO_coord_384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_385 CaMgO_coord_385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_386 CaMgO_coord_386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_387 CaMgO_coord_387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_388 CaMgO_coord_388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_389 CaMgO_coord_389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_390 CaMgO_coord_390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_391 CaMgO_coord_391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_392 CaMgO_coord_392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_393 CaMgO_coord_393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_394 CaMgO_coord_394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_395 CaMgO_coord_395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_396 CaMgO_coord_396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_397 CaMgO_coord_397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_398 CaMgO_coord_398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_399 CaMgO_coord_399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_399


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_400 CaMgO_coord_400
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_400


