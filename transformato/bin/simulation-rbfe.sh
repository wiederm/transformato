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

run_complex () {
input=lig_in_complex
init=lig_in_complex
pstep=lig_in_complex
istep=lig_in_complex
irst=lig_in_complex
orst=lig_in_complex_rst
python -u openmm_run.py -i ${input}.inp -t toppar.str -p ${init}.psf -c ${init}.crd -irst ${irst}.rst -orst ${irst} -odcd ${istep}.dcd &> complex_out.log 
}

run_waterbox () {
input=lig_in_waterbox
init=lig_in_waterbox
pstep=lig_in_waterbox
istep=lig_in_waterbox
irst=lig_in_waterbox
orst=lig_in_waterbox_rst
python -u openmm_run.py -i ${input}.inp -t toppar.str -p ${init}.psf -c ${init}.crd -irst ${irst}.rst -orst ${irst} -odcd ${istep}.dcd  &> waterbox_out.log 
}

case ${SWITCH} in
1)
run_waterbox
;;
2)
run_waterbox
run_complex
;;
esac
