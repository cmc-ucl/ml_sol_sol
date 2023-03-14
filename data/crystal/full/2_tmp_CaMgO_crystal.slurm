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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_100 CaMgO_full_tmp_100 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_101 CaMgO_full_tmp_101 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_102 CaMgO_full_tmp_102 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_103 CaMgO_full_tmp_103 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_104 CaMgO_full_tmp_104 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_105 CaMgO_full_tmp_105 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_106 CaMgO_full_tmp_106 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_107 CaMgO_full_tmp_107 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_108 CaMgO_full_tmp_108 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_109 CaMgO_full_tmp_109 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_110 CaMgO_full_tmp_110 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_111 CaMgO_full_tmp_111 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_112 CaMgO_full_tmp_112 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_113 CaMgO_full_tmp_113 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_114 CaMgO_full_tmp_114 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_115 CaMgO_full_tmp_115 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_116 CaMgO_full_tmp_116 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_117 CaMgO_full_tmp_117 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_118 CaMgO_full_tmp_118 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_119 CaMgO_full_tmp_119 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_120 CaMgO_full_tmp_120 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_121 CaMgO_full_tmp_121 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_122 CaMgO_full_tmp_122 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_123 CaMgO_full_tmp_123 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_124 CaMgO_full_tmp_124 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_125 CaMgO_full_tmp_125 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_126 CaMgO_full_tmp_126 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_127 CaMgO_full_tmp_127 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_128 CaMgO_full_tmp_128 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_129 CaMgO_full_tmp_129 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_130 CaMgO_full_tmp_130 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_131 CaMgO_full_tmp_131 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_132 CaMgO_full_tmp_132 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_133 CaMgO_full_tmp_133 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_134 CaMgO_full_tmp_134 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_135 CaMgO_full_tmp_135 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_136 CaMgO_full_tmp_136 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_137 CaMgO_full_tmp_137 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_138 CaMgO_full_tmp_138 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_139 CaMgO_full_tmp_139 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_140 CaMgO_full_tmp_140 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_141 CaMgO_full_tmp_141 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_142 CaMgO_full_tmp_142 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_143 CaMgO_full_tmp_143 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_144 CaMgO_full_tmp_144 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_145 CaMgO_full_tmp_145 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_146 CaMgO_full_tmp_146 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_147 CaMgO_full_tmp_147 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_148 CaMgO_full_tmp_148 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_149 CaMgO_full_tmp_149 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_150 CaMgO_full_tmp_150 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_151 CaMgO_full_tmp_151 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_152 CaMgO_full_tmp_152 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_153 CaMgO_full_tmp_153 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_154 CaMgO_full_tmp_154 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_155 CaMgO_full_tmp_155 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_156 CaMgO_full_tmp_156 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_157 CaMgO_full_tmp_157 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_158 CaMgO_full_tmp_158 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_159 CaMgO_full_tmp_159 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_160 CaMgO_full_tmp_160 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_161 CaMgO_full_tmp_161 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_162 CaMgO_full_tmp_162 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_163 CaMgO_full_tmp_163 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_164 CaMgO_full_tmp_164 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_165 CaMgO_full_tmp_165 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_166 CaMgO_full_tmp_166 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_167 CaMgO_full_tmp_167 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_168 CaMgO_full_tmp_168 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_169 CaMgO_full_tmp_169 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_170 CaMgO_full_tmp_170 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_171 CaMgO_full_tmp_171 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_172 CaMgO_full_tmp_172 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_173 CaMgO_full_tmp_173 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_174 CaMgO_full_tmp_174 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_175 CaMgO_full_tmp_175 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_176 CaMgO_full_tmp_176 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_177 CaMgO_full_tmp_177 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_178 CaMgO_full_tmp_178 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_179 CaMgO_full_tmp_179 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_180 CaMgO_full_tmp_180 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_181 CaMgO_full_tmp_181 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_182 CaMgO_full_tmp_182 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_183 CaMgO_full_tmp_183 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_184 CaMgO_full_tmp_184 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_185 CaMgO_full_tmp_185 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_186 CaMgO_full_tmp_186 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_187 CaMgO_full_tmp_187 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_188 CaMgO_full_tmp_188 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_189 CaMgO_full_tmp_189 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_190 CaMgO_full_tmp_190 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_191 CaMgO_full_tmp_191 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_192 CaMgO_full_tmp_192 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_193 CaMgO_full_tmp_193 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_194 CaMgO_full_tmp_194 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_195 CaMgO_full_tmp_195 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_196 CaMgO_full_tmp_196 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_197 CaMgO_full_tmp_197 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_198 CaMgO_full_tmp_198 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_199 CaMgO_full_tmp_199 &

wait
mv CaMgO_full_tmp_100.f9 CaMgO_full_100.f9
mv CaMgO_full_tmp_100.gui CaMgO_full_100.gui
mv CaMgO_full_tmp_101.f9 CaMgO_full_101.f9
mv CaMgO_full_tmp_101.gui CaMgO_full_101.gui
mv CaMgO_full_tmp_102.f9 CaMgO_full_102.f9
mv CaMgO_full_tmp_102.gui CaMgO_full_102.gui
mv CaMgO_full_tmp_103.f9 CaMgO_full_103.f9
mv CaMgO_full_tmp_103.gui CaMgO_full_103.gui
mv CaMgO_full_tmp_104.f9 CaMgO_full_104.f9
mv CaMgO_full_tmp_104.gui CaMgO_full_104.gui
mv CaMgO_full_tmp_105.f9 CaMgO_full_105.f9
mv CaMgO_full_tmp_105.gui CaMgO_full_105.gui
mv CaMgO_full_tmp_106.f9 CaMgO_full_106.f9
mv CaMgO_full_tmp_106.gui CaMgO_full_106.gui
mv CaMgO_full_tmp_107.f9 CaMgO_full_107.f9
mv CaMgO_full_tmp_107.gui CaMgO_full_107.gui
mv CaMgO_full_tmp_108.f9 CaMgO_full_108.f9
mv CaMgO_full_tmp_108.gui CaMgO_full_108.gui
mv CaMgO_full_tmp_109.f9 CaMgO_full_109.f9
mv CaMgO_full_tmp_109.gui CaMgO_full_109.gui
mv CaMgO_full_tmp_110.f9 CaMgO_full_110.f9
mv CaMgO_full_tmp_110.gui CaMgO_full_110.gui
mv CaMgO_full_tmp_111.f9 CaMgO_full_111.f9
mv CaMgO_full_tmp_111.gui CaMgO_full_111.gui
mv CaMgO_full_tmp_112.f9 CaMgO_full_112.f9
mv CaMgO_full_tmp_112.gui CaMgO_full_112.gui
mv CaMgO_full_tmp_113.f9 CaMgO_full_113.f9
mv CaMgO_full_tmp_113.gui CaMgO_full_113.gui
mv CaMgO_full_tmp_114.f9 CaMgO_full_114.f9
mv CaMgO_full_tmp_114.gui CaMgO_full_114.gui
mv CaMgO_full_tmp_115.f9 CaMgO_full_115.f9
mv CaMgO_full_tmp_115.gui CaMgO_full_115.gui
mv CaMgO_full_tmp_116.f9 CaMgO_full_116.f9
mv CaMgO_full_tmp_116.gui CaMgO_full_116.gui
mv CaMgO_full_tmp_117.f9 CaMgO_full_117.f9
mv CaMgO_full_tmp_117.gui CaMgO_full_117.gui
mv CaMgO_full_tmp_118.f9 CaMgO_full_118.f9
mv CaMgO_full_tmp_118.gui CaMgO_full_118.gui
mv CaMgO_full_tmp_119.f9 CaMgO_full_119.f9
mv CaMgO_full_tmp_119.gui CaMgO_full_119.gui
mv CaMgO_full_tmp_120.f9 CaMgO_full_120.f9
mv CaMgO_full_tmp_120.gui CaMgO_full_120.gui
mv CaMgO_full_tmp_121.f9 CaMgO_full_121.f9
mv CaMgO_full_tmp_121.gui CaMgO_full_121.gui
mv CaMgO_full_tmp_122.f9 CaMgO_full_122.f9
mv CaMgO_full_tmp_122.gui CaMgO_full_122.gui
mv CaMgO_full_tmp_123.f9 CaMgO_full_123.f9
mv CaMgO_full_tmp_123.gui CaMgO_full_123.gui
mv CaMgO_full_tmp_124.f9 CaMgO_full_124.f9
mv CaMgO_full_tmp_124.gui CaMgO_full_124.gui
mv CaMgO_full_tmp_125.f9 CaMgO_full_125.f9
mv CaMgO_full_tmp_125.gui CaMgO_full_125.gui
mv CaMgO_full_tmp_126.f9 CaMgO_full_126.f9
mv CaMgO_full_tmp_126.gui CaMgO_full_126.gui
mv CaMgO_full_tmp_127.f9 CaMgO_full_127.f9
mv CaMgO_full_tmp_127.gui CaMgO_full_127.gui
mv CaMgO_full_tmp_128.f9 CaMgO_full_128.f9
mv CaMgO_full_tmp_128.gui CaMgO_full_128.gui
mv CaMgO_full_tmp_129.f9 CaMgO_full_129.f9
mv CaMgO_full_tmp_129.gui CaMgO_full_129.gui
mv CaMgO_full_tmp_130.f9 CaMgO_full_130.f9
mv CaMgO_full_tmp_130.gui CaMgO_full_130.gui
mv CaMgO_full_tmp_131.f9 CaMgO_full_131.f9
mv CaMgO_full_tmp_131.gui CaMgO_full_131.gui
mv CaMgO_full_tmp_132.f9 CaMgO_full_132.f9
mv CaMgO_full_tmp_132.gui CaMgO_full_132.gui
mv CaMgO_full_tmp_133.f9 CaMgO_full_133.f9
mv CaMgO_full_tmp_133.gui CaMgO_full_133.gui
mv CaMgO_full_tmp_134.f9 CaMgO_full_134.f9
mv CaMgO_full_tmp_134.gui CaMgO_full_134.gui
mv CaMgO_full_tmp_135.f9 CaMgO_full_135.f9
mv CaMgO_full_tmp_135.gui CaMgO_full_135.gui
mv CaMgO_full_tmp_136.f9 CaMgO_full_136.f9
mv CaMgO_full_tmp_136.gui CaMgO_full_136.gui
mv CaMgO_full_tmp_137.f9 CaMgO_full_137.f9
mv CaMgO_full_tmp_137.gui CaMgO_full_137.gui
mv CaMgO_full_tmp_138.f9 CaMgO_full_138.f9
mv CaMgO_full_tmp_138.gui CaMgO_full_138.gui
mv CaMgO_full_tmp_139.f9 CaMgO_full_139.f9
mv CaMgO_full_tmp_139.gui CaMgO_full_139.gui
mv CaMgO_full_tmp_140.f9 CaMgO_full_140.f9
mv CaMgO_full_tmp_140.gui CaMgO_full_140.gui
mv CaMgO_full_tmp_141.f9 CaMgO_full_141.f9
mv CaMgO_full_tmp_141.gui CaMgO_full_141.gui
mv CaMgO_full_tmp_142.f9 CaMgO_full_142.f9
mv CaMgO_full_tmp_142.gui CaMgO_full_142.gui
mv CaMgO_full_tmp_143.f9 CaMgO_full_143.f9
mv CaMgO_full_tmp_143.gui CaMgO_full_143.gui
mv CaMgO_full_tmp_144.f9 CaMgO_full_144.f9
mv CaMgO_full_tmp_144.gui CaMgO_full_144.gui
mv CaMgO_full_tmp_145.f9 CaMgO_full_145.f9
mv CaMgO_full_tmp_145.gui CaMgO_full_145.gui
mv CaMgO_full_tmp_146.f9 CaMgO_full_146.f9
mv CaMgO_full_tmp_146.gui CaMgO_full_146.gui
mv CaMgO_full_tmp_147.f9 CaMgO_full_147.f9
mv CaMgO_full_tmp_147.gui CaMgO_full_147.gui
mv CaMgO_full_tmp_148.f9 CaMgO_full_148.f9
mv CaMgO_full_tmp_148.gui CaMgO_full_148.gui
mv CaMgO_full_tmp_149.f9 CaMgO_full_149.f9
mv CaMgO_full_tmp_149.gui CaMgO_full_149.gui
mv CaMgO_full_tmp_150.f9 CaMgO_full_150.f9
mv CaMgO_full_tmp_150.gui CaMgO_full_150.gui
mv CaMgO_full_tmp_151.f9 CaMgO_full_151.f9
mv CaMgO_full_tmp_151.gui CaMgO_full_151.gui
mv CaMgO_full_tmp_152.f9 CaMgO_full_152.f9
mv CaMgO_full_tmp_152.gui CaMgO_full_152.gui
mv CaMgO_full_tmp_153.f9 CaMgO_full_153.f9
mv CaMgO_full_tmp_153.gui CaMgO_full_153.gui
mv CaMgO_full_tmp_154.f9 CaMgO_full_154.f9
mv CaMgO_full_tmp_154.gui CaMgO_full_154.gui
mv CaMgO_full_tmp_155.f9 CaMgO_full_155.f9
mv CaMgO_full_tmp_155.gui CaMgO_full_155.gui
mv CaMgO_full_tmp_156.f9 CaMgO_full_156.f9
mv CaMgO_full_tmp_156.gui CaMgO_full_156.gui
mv CaMgO_full_tmp_157.f9 CaMgO_full_157.f9
mv CaMgO_full_tmp_157.gui CaMgO_full_157.gui
mv CaMgO_full_tmp_158.f9 CaMgO_full_158.f9
mv CaMgO_full_tmp_158.gui CaMgO_full_158.gui
mv CaMgO_full_tmp_159.f9 CaMgO_full_159.f9
mv CaMgO_full_tmp_159.gui CaMgO_full_159.gui
mv CaMgO_full_tmp_160.f9 CaMgO_full_160.f9
mv CaMgO_full_tmp_160.gui CaMgO_full_160.gui
mv CaMgO_full_tmp_161.f9 CaMgO_full_161.f9
mv CaMgO_full_tmp_161.gui CaMgO_full_161.gui
mv CaMgO_full_tmp_162.f9 CaMgO_full_162.f9
mv CaMgO_full_tmp_162.gui CaMgO_full_162.gui
mv CaMgO_full_tmp_163.f9 CaMgO_full_163.f9
mv CaMgO_full_tmp_163.gui CaMgO_full_163.gui
mv CaMgO_full_tmp_164.f9 CaMgO_full_164.f9
mv CaMgO_full_tmp_164.gui CaMgO_full_164.gui
mv CaMgO_full_tmp_165.f9 CaMgO_full_165.f9
mv CaMgO_full_tmp_165.gui CaMgO_full_165.gui
mv CaMgO_full_tmp_166.f9 CaMgO_full_166.f9
mv CaMgO_full_tmp_166.gui CaMgO_full_166.gui
mv CaMgO_full_tmp_167.f9 CaMgO_full_167.f9
mv CaMgO_full_tmp_167.gui CaMgO_full_167.gui
mv CaMgO_full_tmp_168.f9 CaMgO_full_168.f9
mv CaMgO_full_tmp_168.gui CaMgO_full_168.gui
mv CaMgO_full_tmp_169.f9 CaMgO_full_169.f9
mv CaMgO_full_tmp_169.gui CaMgO_full_169.gui
mv CaMgO_full_tmp_170.f9 CaMgO_full_170.f9
mv CaMgO_full_tmp_170.gui CaMgO_full_170.gui
mv CaMgO_full_tmp_171.f9 CaMgO_full_171.f9
mv CaMgO_full_tmp_171.gui CaMgO_full_171.gui
mv CaMgO_full_tmp_172.f9 CaMgO_full_172.f9
mv CaMgO_full_tmp_172.gui CaMgO_full_172.gui
mv CaMgO_full_tmp_173.f9 CaMgO_full_173.f9
mv CaMgO_full_tmp_173.gui CaMgO_full_173.gui
mv CaMgO_full_tmp_174.f9 CaMgO_full_174.f9
mv CaMgO_full_tmp_174.gui CaMgO_full_174.gui
mv CaMgO_full_tmp_175.f9 CaMgO_full_175.f9
mv CaMgO_full_tmp_175.gui CaMgO_full_175.gui
mv CaMgO_full_tmp_176.f9 CaMgO_full_176.f9
mv CaMgO_full_tmp_176.gui CaMgO_full_176.gui
mv CaMgO_full_tmp_177.f9 CaMgO_full_177.f9
mv CaMgO_full_tmp_177.gui CaMgO_full_177.gui
mv CaMgO_full_tmp_178.f9 CaMgO_full_178.f9
mv CaMgO_full_tmp_178.gui CaMgO_full_178.gui
mv CaMgO_full_tmp_179.f9 CaMgO_full_179.f9
mv CaMgO_full_tmp_179.gui CaMgO_full_179.gui
mv CaMgO_full_tmp_180.f9 CaMgO_full_180.f9
mv CaMgO_full_tmp_180.gui CaMgO_full_180.gui
mv CaMgO_full_tmp_181.f9 CaMgO_full_181.f9
mv CaMgO_full_tmp_181.gui CaMgO_full_181.gui
mv CaMgO_full_tmp_182.f9 CaMgO_full_182.f9
mv CaMgO_full_tmp_182.gui CaMgO_full_182.gui
mv CaMgO_full_tmp_183.f9 CaMgO_full_183.f9
mv CaMgO_full_tmp_183.gui CaMgO_full_183.gui
mv CaMgO_full_tmp_184.f9 CaMgO_full_184.f9
mv CaMgO_full_tmp_184.gui CaMgO_full_184.gui
mv CaMgO_full_tmp_185.f9 CaMgO_full_185.f9
mv CaMgO_full_tmp_185.gui CaMgO_full_185.gui
mv CaMgO_full_tmp_186.f9 CaMgO_full_186.f9
mv CaMgO_full_tmp_186.gui CaMgO_full_186.gui
mv CaMgO_full_tmp_187.f9 CaMgO_full_187.f9
mv CaMgO_full_tmp_187.gui CaMgO_full_187.gui
mv CaMgO_full_tmp_188.f9 CaMgO_full_188.f9
mv CaMgO_full_tmp_188.gui CaMgO_full_188.gui
mv CaMgO_full_tmp_189.f9 CaMgO_full_189.f9
mv CaMgO_full_tmp_189.gui CaMgO_full_189.gui
mv CaMgO_full_tmp_190.f9 CaMgO_full_190.f9
mv CaMgO_full_tmp_190.gui CaMgO_full_190.gui
mv CaMgO_full_tmp_191.f9 CaMgO_full_191.f9
mv CaMgO_full_tmp_191.gui CaMgO_full_191.gui
mv CaMgO_full_tmp_192.f9 CaMgO_full_192.f9
mv CaMgO_full_tmp_192.gui CaMgO_full_192.gui
mv CaMgO_full_tmp_193.f9 CaMgO_full_193.f9
mv CaMgO_full_tmp_193.gui CaMgO_full_193.gui
mv CaMgO_full_tmp_194.f9 CaMgO_full_194.f9
mv CaMgO_full_tmp_194.gui CaMgO_full_194.gui
mv CaMgO_full_tmp_195.f9 CaMgO_full_195.f9
mv CaMgO_full_tmp_195.gui CaMgO_full_195.gui
mv CaMgO_full_tmp_196.f9 CaMgO_full_196.f9
mv CaMgO_full_tmp_196.gui CaMgO_full_196.gui
mv CaMgO_full_tmp_197.f9 CaMgO_full_197.f9
mv CaMgO_full_tmp_197.gui CaMgO_full_197.gui
mv CaMgO_full_tmp_198.f9 CaMgO_full_198.f9
mv CaMgO_full_tmp_198.gui CaMgO_full_198.gui
mv CaMgO_full_tmp_199.f9 CaMgO_full_199.f9
mv CaMgO_full_tmp_199.gui CaMgO_full_199.gui
