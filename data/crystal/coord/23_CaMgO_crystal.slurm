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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2201 CaMgO_coord_2201
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2201


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2202 CaMgO_coord_2202
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2202


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2203 CaMgO_coord_2203
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2203


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2204 CaMgO_coord_2204
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2204


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2205 CaMgO_coord_2205
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2205


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2206 CaMgO_coord_2206
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2206


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2207 CaMgO_coord_2207
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2207


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2208 CaMgO_coord_2208
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2208


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2209 CaMgO_coord_2209
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2209


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2210 CaMgO_coord_2210
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2210


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2211 CaMgO_coord_2211
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2211


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2212 CaMgO_coord_2212
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2212


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2213 CaMgO_coord_2213
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2213


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2214 CaMgO_coord_2214
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2214


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2215 CaMgO_coord_2215
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2215


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2216 CaMgO_coord_2216
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2216


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2217 CaMgO_coord_2217
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2217


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2218 CaMgO_coord_2218
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2218


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2219 CaMgO_coord_2219
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2219


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2220 CaMgO_coord_2220
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2220


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2221 CaMgO_coord_2221
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2221


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2222 CaMgO_coord_2222
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2222


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2223 CaMgO_coord_2223
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2223


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2224 CaMgO_coord_2224
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2224


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2225 CaMgO_coord_2225
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2225


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2226 CaMgO_coord_2226
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2226


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2227 CaMgO_coord_2227
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2227


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2228 CaMgO_coord_2228
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2228


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2229 CaMgO_coord_2229
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2229


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2230 CaMgO_coord_2230
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2230


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2231 CaMgO_coord_2231
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2231


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2232 CaMgO_coord_2232
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2232


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2233 CaMgO_coord_2233
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2233


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2234 CaMgO_coord_2234
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2234


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2235 CaMgO_coord_2235
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2235


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2236 CaMgO_coord_2236
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2236


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2237 CaMgO_coord_2237
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2237


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2238 CaMgO_coord_2238
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2238


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2239 CaMgO_coord_2239
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2239


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2240 CaMgO_coord_2240
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2240


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2241 CaMgO_coord_2241
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2241


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2242 CaMgO_coord_2242
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2242


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2243 CaMgO_coord_2243
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2243


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2244 CaMgO_coord_2244
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2244


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2245 CaMgO_coord_2245
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2245


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2246 CaMgO_coord_2246
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2246


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2247 CaMgO_coord_2247
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2247


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2248 CaMgO_coord_2248
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2248


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2249 CaMgO_coord_2249
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2249


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2250 CaMgO_coord_2250
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2250


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2251 CaMgO_coord_2251
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2251


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2252 CaMgO_coord_2252
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2252


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2253 CaMgO_coord_2253
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2253


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2254 CaMgO_coord_2254
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2254


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2255 CaMgO_coord_2255
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2255


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2256 CaMgO_coord_2256
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2256


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2257 CaMgO_coord_2257
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2257


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2258 CaMgO_coord_2258
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2258


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2259 CaMgO_coord_2259
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2259


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2260 CaMgO_coord_2260
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2260


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2261 CaMgO_coord_2261
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2261


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2262 CaMgO_coord_2262
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2262


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2263 CaMgO_coord_2263
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2263


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2264 CaMgO_coord_2264
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2264


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2265 CaMgO_coord_2265
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2265


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2266 CaMgO_coord_2266
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2266


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2267 CaMgO_coord_2267
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2267


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2268 CaMgO_coord_2268
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2268


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2269 CaMgO_coord_2269
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2269


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2270 CaMgO_coord_2270
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2270


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2271 CaMgO_coord_2271
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2271


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2272 CaMgO_coord_2272
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2272


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2273 CaMgO_coord_2273
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2273


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2274 CaMgO_coord_2274
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2274


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2275 CaMgO_coord_2275
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2275


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2276 CaMgO_coord_2276
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2276


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2277 CaMgO_coord_2277
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2277


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2278 CaMgO_coord_2278
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2278


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2279 CaMgO_coord_2279
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2279


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2280 CaMgO_coord_2280
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2280


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2281 CaMgO_coord_2281
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2281


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2282 CaMgO_coord_2282
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2282


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2283 CaMgO_coord_2283
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2283


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2284 CaMgO_coord_2284
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2284


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2285 CaMgO_coord_2285
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2285


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2286 CaMgO_coord_2286
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2286


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2287 CaMgO_coord_2287
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2287


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2288 CaMgO_coord_2288
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2288


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2289 CaMgO_coord_2289
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2289


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2290 CaMgO_coord_2290
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2290


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2291 CaMgO_coord_2291
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2291


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2292 CaMgO_coord_2292
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2292


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2293 CaMgO_coord_2293
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2293


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2294 CaMgO_coord_2294
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2294


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2295 CaMgO_coord_2295
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2295


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2296 CaMgO_coord_2296
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2296


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2297 CaMgO_coord_2297
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2297


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2298 CaMgO_coord_2298
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2298


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2299 CaMgO_coord_2299
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2299


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2300 CaMgO_coord_2300
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2300


