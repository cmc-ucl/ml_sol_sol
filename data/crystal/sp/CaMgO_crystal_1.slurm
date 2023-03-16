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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_1
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_1


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_2
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_2


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_4
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_4


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_5
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_5


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_6
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_6


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_7
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_7


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_8
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_8


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_9
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_9


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_10
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_10


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_11
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_11


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_12
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_12


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_13
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_13


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_14
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_14


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_15
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_15


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_16
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_16


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_17
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_17


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_18
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_18


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_19
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_19


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_20
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_20


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_21
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_21


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_22
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_22


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_23
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_23


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_24
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_24


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_25
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_25


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_26
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_26


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_27
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_27


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_28
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_28


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_29
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_29


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_30
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_30


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_31
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_31


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_32
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_32


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_33
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_33


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_34
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_34


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_35
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_35


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_36
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_36


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_37
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_37


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_38
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_38


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_39
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_39


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_40
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_40


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_41
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_41


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_42
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_42


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_43
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_43


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_44
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_44


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_45
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_45


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_46
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_46


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_47
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_47


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_48
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_48


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_49
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_49


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_50
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_50


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_51
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_51


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_52
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_52


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_53
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_53


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_54
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_54


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_55
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_55


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_56
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_56


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_57
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_57


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_58
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_58


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_59
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_59


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_60
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_60


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_61
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_61


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_62
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_62


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_63
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_63


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_64
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_64


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_65
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_65


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_66
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_66


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_67
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_67


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_68
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_68


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_69
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_69


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_70
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_70


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_71
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_71


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_72
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_72


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_73
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_73


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_74
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_74


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_75
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_75


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_76
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_76


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_77
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_77


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_78
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_78


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_79
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_79


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_80
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_80


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_81
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_81


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_82
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_82


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_83
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_83


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_84
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_84


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_85
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_85


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_86
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_86


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_87
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_87


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_88
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_88


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_89
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_89


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_90
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_90


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_91
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_91


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_92
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_92


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_93
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_93


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_94
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_94


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_95
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_95


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_96
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_96


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_97
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_97


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_98
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_98


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_99
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_99


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_100
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_100


