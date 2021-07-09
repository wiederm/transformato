#$ -S /bin/bash
#$ -M marcus.wieder@univie.ac.at
#$ -m e
#$ -j y
#$ -p -500
#$ -o /data/shared/projects/SGE_LOG/
#$ -l gpu=1


#. /data/shared/software/python_env/anaconda3/etc/profile.d/conda.sh
#conda activate transformato

path=$1
SWITCH=$2

cd ${path}
pwd
hostname


run_vacuum () {
export OPENMM_CPU_THREADS=1
input=lig_in_vacuum
init=lig_in_vacuum
pstep=lig_in_vacuum
istep=lig_in_vacuum
irst=lig_in_vacuum
orst=lig_in_vacuum_rst
python -u openmm_run_vacuum.py -i ${input}.inp -t toppar.str -p ${init}.psf -c ${init}.crd -orst ${irst}.rst -odcd ${istep}.dcd &> vacuum_out.log
}

run_waterbox () {
input=lig_in_waterbox
init=lig_in_waterbox
pstep=lig_in_waterbox
istep=lig_in_waterbox
irst=lig_in_waterbox
orst=lig_in_waterbox_rst
python -u openmm_run.py -i ${input}.inp -t toppar.str -p ${init}.psf -c ${init}.crd  -orst ${irst}.rst -odcd ${istep}.dcd &> waterbox_out.log
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
