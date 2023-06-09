#!/bin/bash

cd ../molecules
for d in * ; do
	cd $d
	rm -r prelim_energy
	mkdir prelim_energy
	cd prelim_energy

	cp ../../../relax_tools/quick_energy/* .

	sed -i "s/frame.xyz/${d}.xyz/g;s/molecule_name/${d}/g" geo-opt.inp
	sed -i "s/frame.xyz/${d}.xyz/g" cp2k.job
	sed -i "s/frame.xyz/${d}.xyz/g" run-cp2k.sh

	sbatch cp2k.job

	cd ..
	cd ..
done
