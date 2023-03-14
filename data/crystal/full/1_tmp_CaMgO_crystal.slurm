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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_0 CaMgO_full_tmp_0 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_1 CaMgO_full_tmp_1 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_2 CaMgO_full_tmp_2 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_3 CaMgO_full_tmp_3 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_4 CaMgO_full_tmp_4 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_5 CaMgO_full_tmp_5 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_6 CaMgO_full_tmp_6 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_7 CaMgO_full_tmp_7 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_8 CaMgO_full_tmp_8 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_9 CaMgO_full_tmp_9 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_10 CaMgO_full_tmp_10 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_11 CaMgO_full_tmp_11 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_12 CaMgO_full_tmp_12 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_13 CaMgO_full_tmp_13 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_14 CaMgO_full_tmp_14 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_15 CaMgO_full_tmp_15 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_16 CaMgO_full_tmp_16 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_17 CaMgO_full_tmp_17 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_18 CaMgO_full_tmp_18 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_19 CaMgO_full_tmp_19 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_20 CaMgO_full_tmp_20 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_21 CaMgO_full_tmp_21 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_22 CaMgO_full_tmp_22 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_23 CaMgO_full_tmp_23 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_24 CaMgO_full_tmp_24 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_25 CaMgO_full_tmp_25 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_26 CaMgO_full_tmp_26 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_27 CaMgO_full_tmp_27 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_28 CaMgO_full_tmp_28 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_29 CaMgO_full_tmp_29 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_30 CaMgO_full_tmp_30 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_31 CaMgO_full_tmp_31 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_32 CaMgO_full_tmp_32 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_33 CaMgO_full_tmp_33 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_34 CaMgO_full_tmp_34 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_35 CaMgO_full_tmp_35 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_36 CaMgO_full_tmp_36 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_37 CaMgO_full_tmp_37 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_38 CaMgO_full_tmp_38 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_39 CaMgO_full_tmp_39 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_40 CaMgO_full_tmp_40 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_41 CaMgO_full_tmp_41 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_42 CaMgO_full_tmp_42 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_43 CaMgO_full_tmp_43 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_44 CaMgO_full_tmp_44 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_45 CaMgO_full_tmp_45 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_46 CaMgO_full_tmp_46 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_47 CaMgO_full_tmp_47 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_48 CaMgO_full_tmp_48 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_49 CaMgO_full_tmp_49 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_50 CaMgO_full_tmp_50 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_51 CaMgO_full_tmp_51 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_52 CaMgO_full_tmp_52 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_53 CaMgO_full_tmp_53 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_54 CaMgO_full_tmp_54 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_55 CaMgO_full_tmp_55 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_56 CaMgO_full_tmp_56 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_57 CaMgO_full_tmp_57 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_58 CaMgO_full_tmp_58 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_59 CaMgO_full_tmp_59 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_60 CaMgO_full_tmp_60 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_61 CaMgO_full_tmp_61 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_62 CaMgO_full_tmp_62 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_63 CaMgO_full_tmp_63 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_64 CaMgO_full_tmp_64 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_65 CaMgO_full_tmp_65 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_66 CaMgO_full_tmp_66 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_67 CaMgO_full_tmp_67 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_68 CaMgO_full_tmp_68 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_69 CaMgO_full_tmp_69 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_70 CaMgO_full_tmp_70 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_71 CaMgO_full_tmp_71 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_72 CaMgO_full_tmp_72 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_73 CaMgO_full_tmp_73 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_74 CaMgO_full_tmp_74 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_75 CaMgO_full_tmp_75 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_76 CaMgO_full_tmp_76 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_77 CaMgO_full_tmp_77 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_78 CaMgO_full_tmp_78 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_79 CaMgO_full_tmp_79 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_80 CaMgO_full_tmp_80 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_81 CaMgO_full_tmp_81 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_82 CaMgO_full_tmp_82 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_83 CaMgO_full_tmp_83 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_84 CaMgO_full_tmp_84 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_85 CaMgO_full_tmp_85 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_86 CaMgO_full_tmp_86 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_87 CaMgO_full_tmp_87 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_88 CaMgO_full_tmp_88 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_89 CaMgO_full_tmp_89 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_90 CaMgO_full_tmp_90 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_91 CaMgO_full_tmp_91 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_92 CaMgO_full_tmp_92 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_93 CaMgO_full_tmp_93 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_94 CaMgO_full_tmp_94 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_95 CaMgO_full_tmp_95 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_96 CaMgO_full_tmp_96 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_97 CaMgO_full_tmp_97 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_98 CaMgO_full_tmp_98 &

timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi CaMgO_full_tmp_99 CaMgO_full_tmp_99 &

wait
mv CaMgO_full_tmp_0.f9 CaMgO_full_0.f9
mv CaMgO_full_tmp_0.gui CaMgO_full_0.gui
mv CaMgO_full_tmp_1.f9 CaMgO_full_1.f9
mv CaMgO_full_tmp_1.gui CaMgO_full_1.gui
mv CaMgO_full_tmp_2.f9 CaMgO_full_2.f9
mv CaMgO_full_tmp_2.gui CaMgO_full_2.gui
mv CaMgO_full_tmp_3.f9 CaMgO_full_3.f9
mv CaMgO_full_tmp_3.gui CaMgO_full_3.gui
mv CaMgO_full_tmp_4.f9 CaMgO_full_4.f9
mv CaMgO_full_tmp_4.gui CaMgO_full_4.gui
mv CaMgO_full_tmp_5.f9 CaMgO_full_5.f9
mv CaMgO_full_tmp_5.gui CaMgO_full_5.gui
mv CaMgO_full_tmp_6.f9 CaMgO_full_6.f9
mv CaMgO_full_tmp_6.gui CaMgO_full_6.gui
mv CaMgO_full_tmp_7.f9 CaMgO_full_7.f9
mv CaMgO_full_tmp_7.gui CaMgO_full_7.gui
mv CaMgO_full_tmp_8.f9 CaMgO_full_8.f9
mv CaMgO_full_tmp_8.gui CaMgO_full_8.gui
mv CaMgO_full_tmp_9.f9 CaMgO_full_9.f9
mv CaMgO_full_tmp_9.gui CaMgO_full_9.gui
mv CaMgO_full_tmp_10.f9 CaMgO_full_10.f9
mv CaMgO_full_tmp_10.gui CaMgO_full_10.gui
mv CaMgO_full_tmp_11.f9 CaMgO_full_11.f9
mv CaMgO_full_tmp_11.gui CaMgO_full_11.gui
mv CaMgO_full_tmp_12.f9 CaMgO_full_12.f9
mv CaMgO_full_tmp_12.gui CaMgO_full_12.gui
mv CaMgO_full_tmp_13.f9 CaMgO_full_13.f9
mv CaMgO_full_tmp_13.gui CaMgO_full_13.gui
mv CaMgO_full_tmp_14.f9 CaMgO_full_14.f9
mv CaMgO_full_tmp_14.gui CaMgO_full_14.gui
mv CaMgO_full_tmp_15.f9 CaMgO_full_15.f9
mv CaMgO_full_tmp_15.gui CaMgO_full_15.gui
mv CaMgO_full_tmp_16.f9 CaMgO_full_16.f9
mv CaMgO_full_tmp_16.gui CaMgO_full_16.gui
mv CaMgO_full_tmp_17.f9 CaMgO_full_17.f9
mv CaMgO_full_tmp_17.gui CaMgO_full_17.gui
mv CaMgO_full_tmp_18.f9 CaMgO_full_18.f9
mv CaMgO_full_tmp_18.gui CaMgO_full_18.gui
mv CaMgO_full_tmp_19.f9 CaMgO_full_19.f9
mv CaMgO_full_tmp_19.gui CaMgO_full_19.gui
mv CaMgO_full_tmp_20.f9 CaMgO_full_20.f9
mv CaMgO_full_tmp_20.gui CaMgO_full_20.gui
mv CaMgO_full_tmp_21.f9 CaMgO_full_21.f9
mv CaMgO_full_tmp_21.gui CaMgO_full_21.gui
mv CaMgO_full_tmp_22.f9 CaMgO_full_22.f9
mv CaMgO_full_tmp_22.gui CaMgO_full_22.gui
mv CaMgO_full_tmp_23.f9 CaMgO_full_23.f9
mv CaMgO_full_tmp_23.gui CaMgO_full_23.gui
mv CaMgO_full_tmp_24.f9 CaMgO_full_24.f9
mv CaMgO_full_tmp_24.gui CaMgO_full_24.gui
mv CaMgO_full_tmp_25.f9 CaMgO_full_25.f9
mv CaMgO_full_tmp_25.gui CaMgO_full_25.gui
mv CaMgO_full_tmp_26.f9 CaMgO_full_26.f9
mv CaMgO_full_tmp_26.gui CaMgO_full_26.gui
mv CaMgO_full_tmp_27.f9 CaMgO_full_27.f9
mv CaMgO_full_tmp_27.gui CaMgO_full_27.gui
mv CaMgO_full_tmp_28.f9 CaMgO_full_28.f9
mv CaMgO_full_tmp_28.gui CaMgO_full_28.gui
mv CaMgO_full_tmp_29.f9 CaMgO_full_29.f9
mv CaMgO_full_tmp_29.gui CaMgO_full_29.gui
mv CaMgO_full_tmp_30.f9 CaMgO_full_30.f9
mv CaMgO_full_tmp_30.gui CaMgO_full_30.gui
mv CaMgO_full_tmp_31.f9 CaMgO_full_31.f9
mv CaMgO_full_tmp_31.gui CaMgO_full_31.gui
mv CaMgO_full_tmp_32.f9 CaMgO_full_32.f9
mv CaMgO_full_tmp_32.gui CaMgO_full_32.gui
mv CaMgO_full_tmp_33.f9 CaMgO_full_33.f9
mv CaMgO_full_tmp_33.gui CaMgO_full_33.gui
mv CaMgO_full_tmp_34.f9 CaMgO_full_34.f9
mv CaMgO_full_tmp_34.gui CaMgO_full_34.gui
mv CaMgO_full_tmp_35.f9 CaMgO_full_35.f9
mv CaMgO_full_tmp_35.gui CaMgO_full_35.gui
mv CaMgO_full_tmp_36.f9 CaMgO_full_36.f9
mv CaMgO_full_tmp_36.gui CaMgO_full_36.gui
mv CaMgO_full_tmp_37.f9 CaMgO_full_37.f9
mv CaMgO_full_tmp_37.gui CaMgO_full_37.gui
mv CaMgO_full_tmp_38.f9 CaMgO_full_38.f9
mv CaMgO_full_tmp_38.gui CaMgO_full_38.gui
mv CaMgO_full_tmp_39.f9 CaMgO_full_39.f9
mv CaMgO_full_tmp_39.gui CaMgO_full_39.gui
mv CaMgO_full_tmp_40.f9 CaMgO_full_40.f9
mv CaMgO_full_tmp_40.gui CaMgO_full_40.gui
mv CaMgO_full_tmp_41.f9 CaMgO_full_41.f9
mv CaMgO_full_tmp_41.gui CaMgO_full_41.gui
mv CaMgO_full_tmp_42.f9 CaMgO_full_42.f9
mv CaMgO_full_tmp_42.gui CaMgO_full_42.gui
mv CaMgO_full_tmp_43.f9 CaMgO_full_43.f9
mv CaMgO_full_tmp_43.gui CaMgO_full_43.gui
mv CaMgO_full_tmp_44.f9 CaMgO_full_44.f9
mv CaMgO_full_tmp_44.gui CaMgO_full_44.gui
mv CaMgO_full_tmp_45.f9 CaMgO_full_45.f9
mv CaMgO_full_tmp_45.gui CaMgO_full_45.gui
mv CaMgO_full_tmp_46.f9 CaMgO_full_46.f9
mv CaMgO_full_tmp_46.gui CaMgO_full_46.gui
mv CaMgO_full_tmp_47.f9 CaMgO_full_47.f9
mv CaMgO_full_tmp_47.gui CaMgO_full_47.gui
mv CaMgO_full_tmp_48.f9 CaMgO_full_48.f9
mv CaMgO_full_tmp_48.gui CaMgO_full_48.gui
mv CaMgO_full_tmp_49.f9 CaMgO_full_49.f9
mv CaMgO_full_tmp_49.gui CaMgO_full_49.gui
mv CaMgO_full_tmp_50.f9 CaMgO_full_50.f9
mv CaMgO_full_tmp_50.gui CaMgO_full_50.gui
mv CaMgO_full_tmp_51.f9 CaMgO_full_51.f9
mv CaMgO_full_tmp_51.gui CaMgO_full_51.gui
mv CaMgO_full_tmp_52.f9 CaMgO_full_52.f9
mv CaMgO_full_tmp_52.gui CaMgO_full_52.gui
mv CaMgO_full_tmp_53.f9 CaMgO_full_53.f9
mv CaMgO_full_tmp_53.gui CaMgO_full_53.gui
mv CaMgO_full_tmp_54.f9 CaMgO_full_54.f9
mv CaMgO_full_tmp_54.gui CaMgO_full_54.gui
mv CaMgO_full_tmp_55.f9 CaMgO_full_55.f9
mv CaMgO_full_tmp_55.gui CaMgO_full_55.gui
mv CaMgO_full_tmp_56.f9 CaMgO_full_56.f9
mv CaMgO_full_tmp_56.gui CaMgO_full_56.gui
mv CaMgO_full_tmp_57.f9 CaMgO_full_57.f9
mv CaMgO_full_tmp_57.gui CaMgO_full_57.gui
mv CaMgO_full_tmp_58.f9 CaMgO_full_58.f9
mv CaMgO_full_tmp_58.gui CaMgO_full_58.gui
mv CaMgO_full_tmp_59.f9 CaMgO_full_59.f9
mv CaMgO_full_tmp_59.gui CaMgO_full_59.gui
mv CaMgO_full_tmp_60.f9 CaMgO_full_60.f9
mv CaMgO_full_tmp_60.gui CaMgO_full_60.gui
mv CaMgO_full_tmp_61.f9 CaMgO_full_61.f9
mv CaMgO_full_tmp_61.gui CaMgO_full_61.gui
mv CaMgO_full_tmp_62.f9 CaMgO_full_62.f9
mv CaMgO_full_tmp_62.gui CaMgO_full_62.gui
mv CaMgO_full_tmp_63.f9 CaMgO_full_63.f9
mv CaMgO_full_tmp_63.gui CaMgO_full_63.gui
mv CaMgO_full_tmp_64.f9 CaMgO_full_64.f9
mv CaMgO_full_tmp_64.gui CaMgO_full_64.gui
mv CaMgO_full_tmp_65.f9 CaMgO_full_65.f9
mv CaMgO_full_tmp_65.gui CaMgO_full_65.gui
mv CaMgO_full_tmp_66.f9 CaMgO_full_66.f9
mv CaMgO_full_tmp_66.gui CaMgO_full_66.gui
mv CaMgO_full_tmp_67.f9 CaMgO_full_67.f9
mv CaMgO_full_tmp_67.gui CaMgO_full_67.gui
mv CaMgO_full_tmp_68.f9 CaMgO_full_68.f9
mv CaMgO_full_tmp_68.gui CaMgO_full_68.gui
mv CaMgO_full_tmp_69.f9 CaMgO_full_69.f9
mv CaMgO_full_tmp_69.gui CaMgO_full_69.gui
mv CaMgO_full_tmp_70.f9 CaMgO_full_70.f9
mv CaMgO_full_tmp_70.gui CaMgO_full_70.gui
mv CaMgO_full_tmp_71.f9 CaMgO_full_71.f9
mv CaMgO_full_tmp_71.gui CaMgO_full_71.gui
mv CaMgO_full_tmp_72.f9 CaMgO_full_72.f9
mv CaMgO_full_tmp_72.gui CaMgO_full_72.gui
mv CaMgO_full_tmp_73.f9 CaMgO_full_73.f9
mv CaMgO_full_tmp_73.gui CaMgO_full_73.gui
mv CaMgO_full_tmp_74.f9 CaMgO_full_74.f9
mv CaMgO_full_tmp_74.gui CaMgO_full_74.gui
mv CaMgO_full_tmp_75.f9 CaMgO_full_75.f9
mv CaMgO_full_tmp_75.gui CaMgO_full_75.gui
mv CaMgO_full_tmp_76.f9 CaMgO_full_76.f9
mv CaMgO_full_tmp_76.gui CaMgO_full_76.gui
mv CaMgO_full_tmp_77.f9 CaMgO_full_77.f9
mv CaMgO_full_tmp_77.gui CaMgO_full_77.gui
mv CaMgO_full_tmp_78.f9 CaMgO_full_78.f9
mv CaMgO_full_tmp_78.gui CaMgO_full_78.gui
mv CaMgO_full_tmp_79.f9 CaMgO_full_79.f9
mv CaMgO_full_tmp_79.gui CaMgO_full_79.gui
mv CaMgO_full_tmp_80.f9 CaMgO_full_80.f9
mv CaMgO_full_tmp_80.gui CaMgO_full_80.gui
mv CaMgO_full_tmp_81.f9 CaMgO_full_81.f9
mv CaMgO_full_tmp_81.gui CaMgO_full_81.gui
mv CaMgO_full_tmp_82.f9 CaMgO_full_82.f9
mv CaMgO_full_tmp_82.gui CaMgO_full_82.gui
mv CaMgO_full_tmp_83.f9 CaMgO_full_83.f9
mv CaMgO_full_tmp_83.gui CaMgO_full_83.gui
mv CaMgO_full_tmp_84.f9 CaMgO_full_84.f9
mv CaMgO_full_tmp_84.gui CaMgO_full_84.gui
mv CaMgO_full_tmp_85.f9 CaMgO_full_85.f9
mv CaMgO_full_tmp_85.gui CaMgO_full_85.gui
mv CaMgO_full_tmp_86.f9 CaMgO_full_86.f9
mv CaMgO_full_tmp_86.gui CaMgO_full_86.gui
mv CaMgO_full_tmp_87.f9 CaMgO_full_87.f9
mv CaMgO_full_tmp_87.gui CaMgO_full_87.gui
mv CaMgO_full_tmp_88.f9 CaMgO_full_88.f9
mv CaMgO_full_tmp_88.gui CaMgO_full_88.gui
mv CaMgO_full_tmp_89.f9 CaMgO_full_89.f9
mv CaMgO_full_tmp_89.gui CaMgO_full_89.gui
mv CaMgO_full_tmp_90.f9 CaMgO_full_90.f9
mv CaMgO_full_tmp_90.gui CaMgO_full_90.gui
mv CaMgO_full_tmp_91.f9 CaMgO_full_91.f9
mv CaMgO_full_tmp_91.gui CaMgO_full_91.gui
mv CaMgO_full_tmp_92.f9 CaMgO_full_92.f9
mv CaMgO_full_tmp_92.gui CaMgO_full_92.gui
mv CaMgO_full_tmp_93.f9 CaMgO_full_93.f9
mv CaMgO_full_tmp_93.gui CaMgO_full_93.gui
mv CaMgO_full_tmp_94.f9 CaMgO_full_94.f9
mv CaMgO_full_tmp_94.gui CaMgO_full_94.gui
mv CaMgO_full_tmp_95.f9 CaMgO_full_95.f9
mv CaMgO_full_tmp_95.gui CaMgO_full_95.gui
mv CaMgO_full_tmp_96.f9 CaMgO_full_96.f9
mv CaMgO_full_tmp_96.gui CaMgO_full_96.gui
mv CaMgO_full_tmp_97.f9 CaMgO_full_97.f9
mv CaMgO_full_tmp_97.gui CaMgO_full_97.gui
mv CaMgO_full_tmp_98.f9 CaMgO_full_98.f9
mv CaMgO_full_tmp_98.gui CaMgO_full_98.gui
mv CaMgO_full_tmp_99.f9 CaMgO_full_99.f9
mv CaMgO_full_tmp_99.gui CaMgO_full_99.gui
