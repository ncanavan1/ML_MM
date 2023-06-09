#!/bin/bash

while read p; do
	cd ../molecules/$p/geometry_optimisation

        sed -i "s/frame.xyz/${p}.xyz/g;s/molecule_name/${p}/g" geo-opt-restart.inp
        sed -i "s/frame.xyz/${p}.xyz/g" cp2k-restart.job
        sed -i "s/frame.xyz/${p}.xyz/g" run-restart.sh

	sbatch cp2k-restart.job

	cd ../../../

done < unfinished.txt
