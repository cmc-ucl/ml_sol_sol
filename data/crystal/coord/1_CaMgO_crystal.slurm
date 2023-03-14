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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_1 CaMgO_coord_1
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_1


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_2 CaMgO_coord_2
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_2


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3 CaMgO_coord_3
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_4 CaMgO_coord_4
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_4


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_5 CaMgO_coord_5
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_5


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_6 CaMgO_coord_6
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_6


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_7 CaMgO_coord_7
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_7


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_8 CaMgO_coord_8
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_8


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_9 CaMgO_coord_9
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_9


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_10 CaMgO_coord_10
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_10


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_11 CaMgO_coord_11
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_11


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_12 CaMgO_coord_12
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_12


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_13 CaMgO_coord_13
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_13


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_14 CaMgO_coord_14
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_14


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_15 CaMgO_coord_15
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_15


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_16 CaMgO_coord_16
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_16


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_17 CaMgO_coord_17
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_17


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_18 CaMgO_coord_18
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_18


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_19 CaMgO_coord_19
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_19


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_20 CaMgO_coord_20
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_20


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_21 CaMgO_coord_21
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_21


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_22 CaMgO_coord_22
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_22


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_23 CaMgO_coord_23
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_23


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_24 CaMgO_coord_24
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_24


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_25 CaMgO_coord_25
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_25


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_26 CaMgO_coord_26
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_26


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_27 CaMgO_coord_27
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_27


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_28 CaMgO_coord_28
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_28


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_29 CaMgO_coord_29
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_29


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_30 CaMgO_coord_30
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_30


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_31 CaMgO_coord_31
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_31


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_32 CaMgO_coord_32
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_32


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_33 CaMgO_coord_33
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_33


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_34 CaMgO_coord_34
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_34


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_35 CaMgO_coord_35
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_35


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_36 CaMgO_coord_36
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_36


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_37 CaMgO_coord_37
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_37


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_38 CaMgO_coord_38
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_38


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_39 CaMgO_coord_39
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_39


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_40 CaMgO_coord_40
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_40


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_41 CaMgO_coord_41
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_41


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_42 CaMgO_coord_42
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_42


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_43 CaMgO_coord_43
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_43


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_44 CaMgO_coord_44
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_44


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_45 CaMgO_coord_45
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_45


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_46 CaMgO_coord_46
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_46


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_47 CaMgO_coord_47
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_47


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_48 CaMgO_coord_48
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_48


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_49 CaMgO_coord_49
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_49


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_50 CaMgO_coord_50
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_50


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_51 CaMgO_coord_51
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_51


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_52 CaMgO_coord_52
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_52


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_53 CaMgO_coord_53
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_53


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_54 CaMgO_coord_54
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_54


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_55 CaMgO_coord_55
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_55


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_56 CaMgO_coord_56
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_56


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_57 CaMgO_coord_57
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_57


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_58 CaMgO_coord_58
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_58


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_59 CaMgO_coord_59
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_59


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_60 CaMgO_coord_60
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_60


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_61 CaMgO_coord_61
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_61


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_62 CaMgO_coord_62
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_62


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_63 CaMgO_coord_63
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_63


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_64 CaMgO_coord_64
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_64


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_65 CaMgO_coord_65
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_65


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_66 CaMgO_coord_66
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_66


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_67 CaMgO_coord_67
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_67


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_68 CaMgO_coord_68
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_68


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_69 CaMgO_coord_69
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_69


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_70 CaMgO_coord_70
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_70


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_71 CaMgO_coord_71
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_71


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_72 CaMgO_coord_72
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_72


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_73 CaMgO_coord_73
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_73


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_74 CaMgO_coord_74
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_74


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_75 CaMgO_coord_75
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_75


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_76 CaMgO_coord_76
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_76


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_77 CaMgO_coord_77
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_77


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_78 CaMgO_coord_78
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_78


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_79 CaMgO_coord_79
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_79


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_80 CaMgO_coord_80
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_80


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_81 CaMgO_coord_81
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_81


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_82 CaMgO_coord_82
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_82


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_83 CaMgO_coord_83
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_83


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_84 CaMgO_coord_84
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_84


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_85 CaMgO_coord_85
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_85


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_86 CaMgO_coord_86
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_86


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_87 CaMgO_coord_87
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_87


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_88 CaMgO_coord_88
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_88


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_89 CaMgO_coord_89
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_89


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_90 CaMgO_coord_90
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_90


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_91 CaMgO_coord_91
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_91


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_92 CaMgO_coord_92
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_92


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_93 CaMgO_coord_93
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_93


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_94 CaMgO_coord_94
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_94


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_95 CaMgO_coord_95
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_95


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_96 CaMgO_coord_96
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_96


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_97 CaMgO_coord_97
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_97


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_98 CaMgO_coord_98
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_98


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_99 CaMgO_coord_99
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_99


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_100 CaMgO_coord_100
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_100


