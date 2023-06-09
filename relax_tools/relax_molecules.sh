#!/bin/bash


cd ../molecules
for d in * ; do

cd $d   
rm -r geometry_optimisation 
mkdir geometry_optimisation
cd geometry_optimisation


cp ../../../relax_tools/relax_energy/* .

sed -i "s/frame.xyz/${d}.xyz/g" geo-opt-relax.inp
sed -i "s/frame.xyz/${d}.xyz/g" cp2k-relax.job
sed -i "s/frame.xyz/${d}.xyz/g" run-cp2k-relax.sh

sbatch cp2k-relax.job

cd ../..

done
