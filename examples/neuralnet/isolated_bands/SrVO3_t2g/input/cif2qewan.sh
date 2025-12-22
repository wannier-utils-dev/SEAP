#!/bin/bash
#PBS -l nodes=1:ppn=32
#PBS -q GroupD
#PBS -N example_SVO

BAND_DATA="band.dat"
PY_DIR="/path/to/wannierinit"
QE_DIR='/path/to/QuantumESPRESSO' # != ver 7.3
WANNIER90_DIR='/path/to/wannier90'
nproc=32

cd $PBS_O_WORKDIR

python3 ${PY_DIR}/python/cif2qewan/cif2qewan.py *.cif cif2qewan.toml

mpirun -n ${nproc} ${QE_DIR}/bin/pw.x -nk 8 < scf.in > scf.out
cp -r work check_wannier/
cp -r work band/

nbnd_scf=$(cat ./scf.in | grep nbnd | awk '{print $3}')
sed -i "s/nbnd .*/nbnd = ${nbnd_scf}/g" nscf.in

#
# wannierinit
#
mkdir pred
cd pred
 
python3 -m seap.core.select_bands ../ --man $BAND_DATA --bl 1.5
tail -n +2 $BAND_DATA | while read -r nb; do
    mpirun -n  1 ${QE_DIR}/bin/pp.x < pwscf.pp_${nb}.in > pwscf.pp_${nb}.out
done

python3 -m seap.core.postppx --bl 1.5 --cl 1.5
python3 -m seap.core.predict --mode nn --nnid 1
python3 -m seap.core.proj

cd ../

nbnd=$(cat ./pred/nbnd | awk '{print $1}')
sed -i "s/nbnd .*/nbnd = ${nbnd}/g" nscf.in
python3 -m seap.core.mod_win

#
#
#

echo 'nscf'
mpirun -n ${nproc} ${QE_DIR}/bin/pw.x -nk 8 < nscf.in > nscf.out

echo 'w90-pp'
mpirun -n  1 ${WANNIER90_DIR}/wannier90.x -pp pwscf

echo 'pw2wan'
mpirun -n  1 ${QE_DIR}/bin/pw2wannier90.x < pw2wan.in > pw2wan.out

rm -r work

echo 'set dis_froz_max = EF+2.0eV'
ef=$(grep Fermi nscf.out | sed -e "s/[^0-9.]//g")
ef_new=$(bc -l <<< "$ef + 2.0")
sed -i "s/dis_froz_max .*/dis_froz_max = $ef_new/g" pwscf.win
echo 'w90'
mpirun -n 1 ${WANNIER90_DIR}/wannier90.x pwscf

cd check_wannier
sed -i "s/nbnd .*/nbnd = ${nbnd}/g" nscf.in
mpirun -n ${nproc} ${QE_DIR}/bin/pw.x -nk 8 < nscf.in > nscf.out
rm -r work
cd ../
mpirun -n 1 python ${PY_DIR}/python/cif2qewan/partial_wannier_conv.py

cd band
nbnd_new=$(bc -l <<< "$nbnd + 10")
sed -i "s/nbnd .*/nbnd = ${nbnd_new}/g" nscf.in
mpirun -n ${nproc} ${QE_DIR}/bin/pw.x -nk 8 < nscf.in > nscf.out
mpirun -n  1 ${QE_DIR}/bin/bands.x < band.in > band.out
rm -r work
cd ../
mpirun -n 1 python ${PY_DIR}/python/cif2qewan/band_comp.py
