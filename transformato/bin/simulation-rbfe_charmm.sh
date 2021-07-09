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


run_complex () {
input=charmm_run_complex
OMP_NUM_THREADS=8 /home/mwieder/Work/Software/charmm/exec/gnu/charmm -i ${input}.inp > log_complex.out
}

run_waterbox () {
input=charmm_run_waterbox
OMP_NUM_THREADS=8 /home/mwieder/Work/Software/charmm/exec/gnu/charmm -i ${input}.inp > log_solv.out
}


case ${SWITCH} in
1)
run_complex
;;
2)
run_complex
run_waterbox
;;
esac