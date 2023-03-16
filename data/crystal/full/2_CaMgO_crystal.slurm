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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_100 CaMgO_full_100 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_101 CaMgO_full_101 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_102 CaMgO_full_102 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_103 CaMgO_full_103 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_104 CaMgO_full_104 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_105 CaMgO_full_105 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_106 CaMgO_full_106 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_107 CaMgO_full_107 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_108 CaMgO_full_108 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_109 CaMgO_full_109 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_110 CaMgO_full_110 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_111 CaMgO_full_111 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_112 CaMgO_full_112 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_113 CaMgO_full_113 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_114 CaMgO_full_114 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_115 CaMgO_full_115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_116 CaMgO_full_116 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_117 CaMgO_full_117 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_118 CaMgO_full_118 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_119 CaMgO_full_119 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_120 CaMgO_full_120 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_121 CaMgO_full_121 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_122 CaMgO_full_122 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_123 CaMgO_full_123 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_124 CaMgO_full_124 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_125 CaMgO_full_125 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_126 CaMgO_full_126 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_127 CaMgO_full_127 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_128 CaMgO_full_128 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_129 CaMgO_full_129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_130 CaMgO_full_130 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_131 CaMgO_full_131 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_132 CaMgO_full_132 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_133 CaMgO_full_133 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_134 CaMgO_full_134 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_135 CaMgO_full_135 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_136 CaMgO_full_136 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_137 CaMgO_full_137 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_138 CaMgO_full_138 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_139 CaMgO_full_139 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_140 CaMgO_full_140 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_141 CaMgO_full_141 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_142 CaMgO_full_142 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_143 CaMgO_full_143 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_144 CaMgO_full_144 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_145 CaMgO_full_145 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_146 CaMgO_full_146 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_147 CaMgO_full_147 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_148 CaMgO_full_148 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_149 CaMgO_full_149 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_150 CaMgO_full_150 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_151 CaMgO_full_151 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_152 CaMgO_full_152 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_153 CaMgO_full_153 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_154 CaMgO_full_154 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_155 CaMgO_full_155 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_156 CaMgO_full_156 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_157 CaMgO_full_157 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_158 CaMgO_full_158 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_159 CaMgO_full_159 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_160 CaMgO_full_160 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_161 CaMgO_full_161 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_162 CaMgO_full_162 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_163 CaMgO_full_163 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_164 CaMgO_full_164 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_165 CaMgO_full_165 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_166 CaMgO_full_166 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_167 CaMgO_full_167 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_168 CaMgO_full_168 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_169 CaMgO_full_169 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_170 CaMgO_full_170 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_171 CaMgO_full_171 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_172 CaMgO_full_172 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_173 CaMgO_full_173 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_174 CaMgO_full_174 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_175 CaMgO_full_175 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_176 CaMgO_full_176 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_177 CaMgO_full_177 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_178 CaMgO_full_178 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_179 CaMgO_full_179 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_180 CaMgO_full_180 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_181 CaMgO_full_181 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_182 CaMgO_full_182 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_183 CaMgO_full_183 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_184 CaMgO_full_184 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_185 CaMgO_full_185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_186 CaMgO_full_186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_187 CaMgO_full_187 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_188 CaMgO_full_188 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_189 CaMgO_full_189 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_190 CaMgO_full_190 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_191 CaMgO_full_191 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_192 CaMgO_full_192 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_193 CaMgO_full_193 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_194 CaMgO_full_194 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_195 CaMgO_full_195 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_196 CaMgO_full_196 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_197 CaMgO_full_197 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_198 CaMgO_full_198 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_199 CaMgO_full_199 &

wait
