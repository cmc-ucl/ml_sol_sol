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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_300


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_301
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_301


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_302
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_302


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_303
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_303


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_304
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_304


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_305
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_305


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_306
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_306


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_307
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_307


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_308
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_308


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_309
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_309


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_310
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_310


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_311
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_311


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_312
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_312


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_313
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_313


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_314
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_314


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_315
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_315


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_316
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_316


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_317
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_317


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_318
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_318


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_319
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_319


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_320
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_320


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_321
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_321


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_322
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_322


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_323
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_323


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_324
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_324


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_325
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_325


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_326
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_326


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_327
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_327


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_328
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_328


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_329
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_329


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_330
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_330


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_331
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_331


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_332
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_332


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_333
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_333


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_334
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_334


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_335
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_335


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_336
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_336


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_337
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_337


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_338
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_338


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_339
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_339


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_340
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_340


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_341
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_341


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_342
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_342


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_343
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_343


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_344
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_344


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_345
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_345


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_346
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_346


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_347
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_347


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_348
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_348


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_349
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_349


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_350
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_350


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_351
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_351


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_352
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_352


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_353
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_353


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_354
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_354


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_355
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_355


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_356
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_356


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_357
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_357


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_358
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_358


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_359
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_359


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_360
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_360


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_361
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_361


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_362
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_362


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_363
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_363


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_364
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_364


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_365
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_365


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_366
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_366


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_367
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_367


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_368
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_368


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_369
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_369


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_370
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_370


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_371
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_371


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_372
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_372


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_373
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_373


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_374
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_374


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_375
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_375


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_376
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_376


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_377
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_377


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_378
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_378


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_379
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_379


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_380
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_380


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_381
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_381


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_382
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_382


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_383
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_383


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_384
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_384


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_385
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_385


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_386
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_386


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_387
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_387


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_388
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_388


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_389
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_389


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_390
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_390


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_391
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_391


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_392
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_392


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_393
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_393


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_394
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_394


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_395
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_395


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_396
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_396


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_397
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_397


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_398
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_398


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_399
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_399


