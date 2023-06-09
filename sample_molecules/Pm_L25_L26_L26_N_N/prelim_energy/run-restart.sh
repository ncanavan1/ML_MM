#!/bin/bash
export inputframe=${1}
export maxmultiplicity=${2}

for multiplicity in $(seq ${maxmultiplicity} -1 0);do
	if [[ -d "run-${multiplicity}" ]] ; then
		cd run-${multiplicity}
		if [[ $(cat output.out | tail -1) ==  *"DUE TO TIME LIMIT"* ]] ; then
			##Run restarts
			source ~/source/cp2k/tools/toolchain/install/setup
			mpirun -np 121 ~/source/cp2k/exe/local/cp2k.popt geo-opt.inp &> output_restart.out
		fi
		cd ../
	else ; then
 		mkdir -p run-${multiplicity}
 		cd run-${multiplicity}
		cat ../geo-opt.inp | sed "s/MLTP/${multiplicity}/" > geo-opt.inp 
		source ~/source/cp2k/tools/toolchain/install/setup
		mpirun -np 121 ~/source/cp2k/exe/local/cp2k.popt geo-opt.inp &> output.out
		cd ../
	fi
done
