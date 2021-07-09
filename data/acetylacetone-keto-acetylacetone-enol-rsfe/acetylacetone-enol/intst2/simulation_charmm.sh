#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1


path=$1

cd ${path}
pwd
hostname

input=charmm_run_vacuum
/home/mwieder/Work/Software/charmm/exec/gnu/charmm -i ${input}.inp > log_vac.out

input=charmm_run_waterbox
OMP_NUM_THREADS=8 /home/mwieder/Work/Software/charmm/exec/gnu/charmm -i ${input}.inp > log_solv.out
