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
timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3601 CaMgO_coord_3601
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3601


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3602 CaMgO_coord_3602
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3602


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3603 CaMgO_coord_3603
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3603


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3604 CaMgO_coord_3604
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3604


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3605 CaMgO_coord_3605
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3605


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3606 CaMgO_coord_3606
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3606


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3607 CaMgO_coord_3607
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3607


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3608 CaMgO_coord_3608
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3608


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3609 CaMgO_coord_3609
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3609


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3610 CaMgO_coord_3610
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3610


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3611 CaMgO_coord_3611
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3611


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3612 CaMgO_coord_3612
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3612


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3613 CaMgO_coord_3613
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3613


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3614 CaMgO_coord_3614
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3614


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3615 CaMgO_coord_3615
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3615


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3616 CaMgO_coord_3616
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3616


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3617 CaMgO_coord_3617
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3617


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3618 CaMgO_coord_3618
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3618


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3619 CaMgO_coord_3619
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3619


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3620 CaMgO_coord_3620
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3620


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3621 CaMgO_coord_3621
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3621


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3622 CaMgO_coord_3622
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3622


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3623 CaMgO_coord_3623
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3623


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3624 CaMgO_coord_3624
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3624


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3625 CaMgO_coord_3625
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3625


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3626 CaMgO_coord_3626
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3626


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3627 CaMgO_coord_3627
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3627


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3628 CaMgO_coord_3628
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3628


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3629 CaMgO_coord_3629
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3629


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3630 CaMgO_coord_3630
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3630


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3631 CaMgO_coord_3631
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3631


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3632 CaMgO_coord_3632
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3632


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3633 CaMgO_coord_3633
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3633


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3634 CaMgO_coord_3634
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3634


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3635 CaMgO_coord_3635
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3635


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3636 CaMgO_coord_3636
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3636


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3637 CaMgO_coord_3637
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3637


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3638 CaMgO_coord_3638
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3638


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3639 CaMgO_coord_3639
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3639


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3640 CaMgO_coord_3640
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3640


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3641 CaMgO_coord_3641
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3641


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3642 CaMgO_coord_3642
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3642


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3643 CaMgO_coord_3643
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3643


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3644 CaMgO_coord_3644
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3644


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3645 CaMgO_coord_3645
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3645


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3646 CaMgO_coord_3646
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3646


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3647 CaMgO_coord_3647
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3647


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3648 CaMgO_coord_3648
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3648


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3649 CaMgO_coord_3649
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3649


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3650 CaMgO_coord_3650
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3650


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3651 CaMgO_coord_3651
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3651


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3652 CaMgO_coord_3652
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3652


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3653 CaMgO_coord_3653
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3653


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3654 CaMgO_coord_3654
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3654


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3655 CaMgO_coord_3655
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3655


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3656 CaMgO_coord_3656
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3656


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3657 CaMgO_coord_3657
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3657


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3658 CaMgO_coord_3658
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3658


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3659 CaMgO_coord_3659
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3659


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3660 CaMgO_coord_3660
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3660


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3661 CaMgO_coord_3661
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3661


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3662 CaMgO_coord_3662
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3662


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3663 CaMgO_coord_3663
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3663


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3664 CaMgO_coord_3664
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3664


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3665 CaMgO_coord_3665
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3665


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3666 CaMgO_coord_3666
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3666


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3667 CaMgO_coord_3667
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3667


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3668 CaMgO_coord_3668
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3668


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3669 CaMgO_coord_3669
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3669


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3670 CaMgO_coord_3670
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3670


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3671 CaMgO_coord_3671
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3671


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3672 CaMgO_coord_3672
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3672


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3673 CaMgO_coord_3673
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3673


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3674 CaMgO_coord_3674
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3674


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3675 CaMgO_coord_3675
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3675


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3676 CaMgO_coord_3676
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3676


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3677 CaMgO_coord_3677
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3677


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3678 CaMgO_coord_3678
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3678


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3679 CaMgO_coord_3679
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3679


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3680 CaMgO_coord_3680
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3680


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3681 CaMgO_coord_3681
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3681


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3682 CaMgO_coord_3682
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3682


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3683 CaMgO_coord_3683
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3683


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3684 CaMgO_coord_3684
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3684


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3685 CaMgO_coord_3685
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3685


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3686 CaMgO_coord_3686
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3686


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3687 CaMgO_coord_3687
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3687


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3688 CaMgO_coord_3688
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3688


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3689 CaMgO_coord_3689
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3689


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3690 CaMgO_coord_3690
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3690


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3691 CaMgO_coord_3691
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3691


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3692 CaMgO_coord_3692
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3692


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3693 CaMgO_coord_3693
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3693


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3694 CaMgO_coord_3694
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3694


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3695 CaMgO_coord_3695
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3695


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3696 CaMgO_coord_3696
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3696


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3697 CaMgO_coord_3697
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3697


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3698 CaMgO_coord_3698
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3698


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3699 CaMgO_coord_3699
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3699


timeout 2876m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm CaMgO_coord_3700 CaMgO_coord_3700
/work/e05/e05/bcamino/runCRYSTAL/post_proc_slurm crys CaMgO_coord_3700


