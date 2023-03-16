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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_100
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_100


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_101
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_101


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_102
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_102


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_103
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_103


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_104
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_104


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_105
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_105


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_106
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_106


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_107
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_107


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_108
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_108


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_109
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_109


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_110
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_110


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_111
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_111


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_112
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_112


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_113
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_113


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_114
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_114


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_115
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_115


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_116
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_116


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_117
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_117


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_118
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_118


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_119
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_119


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_120
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_120


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_121
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_121


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_122
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_122


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_123
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_123


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_124
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_124


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_125
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_125


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_126
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_126


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_127
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_127


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_128
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_128


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_129
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_129


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_130
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_130


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_131
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_131


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_132
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_132


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_133
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_133


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_134
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_134


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_135
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_135


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_136
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_136


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_137
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_137


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_138
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_138


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_139
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_139


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_140
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_140


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_141
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_141


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_142
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_142


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_143
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_143


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_144
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_144


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_145
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_145


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_146
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_146


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_147
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_147


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_148
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_148


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_149
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_149


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_150
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_150


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_151
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_151


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_152
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_152


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_153
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_153


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_154
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_154


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_155
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_155


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_156
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_156


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_157
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_157


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_158
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_158


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_159
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_159


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_160
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_160


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_161
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_161


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_162
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_162


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_163
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_163


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_164
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_164


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_165
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_165


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_166
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_166


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_167
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_167


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_168
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_168


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_169
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_169


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_170
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_170


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_171
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_171


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_172
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_172


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_173
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_173


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_174
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_174


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_175
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_175


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_176
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_176


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_177
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_177


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_178
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_178


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_179
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_179


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_180
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_180


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_181
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_181


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_182
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_182


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_183
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_183


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_184
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_184


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_185
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_185


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_186
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_186


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_187
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_187


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_188
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_188


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_189
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_189


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_190
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_190


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_191
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_191


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_192
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_192


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_193
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_193


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_194
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_194


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_195
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_195


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_196
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_196


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_197
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_197


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_198
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_198


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_199
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_199


