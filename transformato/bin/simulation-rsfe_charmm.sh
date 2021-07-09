#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1


path=$1
SWITCH=$2

cd ${path}
pwd
hostname


run_vacuum () {
input=charmm_run_vacuum
/home/mwieder/Work/Software/charmm/exec/gnu/charmm -i ${input}.inp > log_vac.out
}

run_waterbox () {
input=charmm_run_waterbox
OMP_NUM_THREADS=8 /home/mwieder/Work/Software/charmm/exec/gnu/charmm -i ${input}.inp > log_solv.out
}


case ${SWITCH} in
1)
run_vacuum
;;
2)
run_vacuum
run_waterbox
;;
esac