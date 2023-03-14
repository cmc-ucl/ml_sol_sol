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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2101
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2101


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2102
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2102


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2103
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2103


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2104
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2104


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2105
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2105


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2106
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2106


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2107
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2107


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2108
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2108


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2109
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2109


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2110
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2110


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2111
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2111


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2112
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2112


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2113
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2113


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2114
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2114


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2115
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2115


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2116
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2116


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2117
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2117


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2118
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2118


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2119
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2119


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2120
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2120


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2121
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2121


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2122
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2122


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2123
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2123


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2124
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2124


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2125
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2125


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2126
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2126


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2127
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2127


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2128
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2128


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2129
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2129


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2130
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2130


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2131
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2131


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2132
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2132


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2133
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2133


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2134
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2134


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2135
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2135


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2136
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2136


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2137
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2137


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2138
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2138


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2139
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2139


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2140
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2140


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2141
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2141


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2142
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2142


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2143
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2143


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2144
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2144


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2145
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2145


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2146
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2146


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2147
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2147


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2148
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2148


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2149
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2149


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2150
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2150


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2151
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2151


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2152
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2152


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2153
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2153


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2154
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2154


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2155
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2155


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2156
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2156


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2157
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2157


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2158
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2158


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2159
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2159


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2160
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2160


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2161
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2161


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2162
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2162


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2163
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2163


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2164
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2164


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2165
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2165


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2166
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2166


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2167
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2167


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2168
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2168


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2169
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2169


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2170
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2170


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2171
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2171


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2172
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2172


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2173
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2173


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2174
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2174


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2175
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2175


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2176
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2176


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2177
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2177


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2178
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2178


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2179
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2179


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2180
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2180


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2181
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2181


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2182
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2182


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2183
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2183


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2184
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2184


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2185
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2185


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2186
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2186


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2187
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2187


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2188
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2188


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2189
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2189


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2190
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2190


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2191
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2191


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2192
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2192


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2193
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2193


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2194
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2194


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2195
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2195


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2196
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2196


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2197
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2197


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2198
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2198


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2199
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2199


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2200


