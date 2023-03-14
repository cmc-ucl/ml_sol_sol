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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3600
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3600


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_sp_3699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_sp_3699


