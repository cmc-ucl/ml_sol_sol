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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_101 CaMgO_coord_101
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_101


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_102 CaMgO_coord_102
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_102


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_103 CaMgO_coord_103
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_103


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_104 CaMgO_coord_104
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_104


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_105 CaMgO_coord_105
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_105


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_106 CaMgO_coord_106
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_106


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_107 CaMgO_coord_107
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_107


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_108 CaMgO_coord_108
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_108


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_109 CaMgO_coord_109
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_109


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_110 CaMgO_coord_110
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_110


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_111 CaMgO_coord_111
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_111


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_112 CaMgO_coord_112
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_112


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_113 CaMgO_coord_113
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_113


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_114 CaMgO_coord_114
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_114


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_115 CaMgO_coord_115
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_115


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_116 CaMgO_coord_116
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_116


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_117 CaMgO_coord_117
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_117


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_118 CaMgO_coord_118
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_118


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_119 CaMgO_coord_119
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_119


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_120 CaMgO_coord_120
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_120


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_121 CaMgO_coord_121
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_121


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_122 CaMgO_coord_122
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_122


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_123 CaMgO_coord_123
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_123


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_124 CaMgO_coord_124
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_124


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_125 CaMgO_coord_125
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_125


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_126 CaMgO_coord_126
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_126


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_127 CaMgO_coord_127
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_127


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_128 CaMgO_coord_128
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_128


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_129 CaMgO_coord_129
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_129


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_130 CaMgO_coord_130
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_130


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_131 CaMgO_coord_131
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_131


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_132 CaMgO_coord_132
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_132


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_133 CaMgO_coord_133
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_133


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_134 CaMgO_coord_134
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_134


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_135 CaMgO_coord_135
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_135


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_136 CaMgO_coord_136
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_136


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_137 CaMgO_coord_137
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_137


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_138 CaMgO_coord_138
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_138


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_139 CaMgO_coord_139
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_139


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_140 CaMgO_coord_140
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_140


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_141 CaMgO_coord_141
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_141


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_142 CaMgO_coord_142
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_142


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_143 CaMgO_coord_143
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_143


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_144 CaMgO_coord_144
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_144


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_145 CaMgO_coord_145
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_145


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_146 CaMgO_coord_146
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_146


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_147 CaMgO_coord_147
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_147


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_148 CaMgO_coord_148
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_148


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_149 CaMgO_coord_149
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_149


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_150 CaMgO_coord_150
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_150


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_151 CaMgO_coord_151
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_151


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_152 CaMgO_coord_152
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_152


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_153 CaMgO_coord_153
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_153


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_154 CaMgO_coord_154
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_154


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_155 CaMgO_coord_155
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_155


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_156 CaMgO_coord_156
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_156


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_157 CaMgO_coord_157
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_157


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_158 CaMgO_coord_158
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_158


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_159 CaMgO_coord_159
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_159


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_160 CaMgO_coord_160
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_160


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_161 CaMgO_coord_161
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_161


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_162 CaMgO_coord_162
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_162


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_163 CaMgO_coord_163
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_163


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_164 CaMgO_coord_164
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_164


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_165 CaMgO_coord_165
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_165


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_166 CaMgO_coord_166
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_166


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_167 CaMgO_coord_167
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_167


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_168 CaMgO_coord_168
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_168


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_169 CaMgO_coord_169
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_169


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_170 CaMgO_coord_170
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_170


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_171 CaMgO_coord_171
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_171


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_172 CaMgO_coord_172
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_172


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_173 CaMgO_coord_173
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_173


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_174 CaMgO_coord_174
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_174


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_175 CaMgO_coord_175
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_175


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_176 CaMgO_coord_176
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_176


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_177 CaMgO_coord_177
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_177


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_178 CaMgO_coord_178
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_178


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_179 CaMgO_coord_179
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_179


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_180 CaMgO_coord_180
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_180


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_181 CaMgO_coord_181
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_181


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_182 CaMgO_coord_182
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_182


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_183 CaMgO_coord_183
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_183


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_184 CaMgO_coord_184
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_184


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_185 CaMgO_coord_185
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_185


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_186 CaMgO_coord_186
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_186


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_187 CaMgO_coord_187
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_187


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_188 CaMgO_coord_188
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_188


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_189 CaMgO_coord_189
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_189


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_190 CaMgO_coord_190
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_190


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_191 CaMgO_coord_191
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_191


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_192 CaMgO_coord_192
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_192


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_193 CaMgO_coord_193
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_193


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_194 CaMgO_coord_194
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_194


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_195 CaMgO_coord_195
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_195


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_196 CaMgO_coord_196
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_196


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_197 CaMgO_coord_197
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_197


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_198 CaMgO_coord_198
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_198


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_199 CaMgO_coord_199
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_199


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_200 CaMgO_coord_200
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_200


