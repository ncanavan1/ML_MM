#!/bin/bash
energy="$PWD/../../prelim_energy_list.txt"
rm $energy
touch $energy
cd ../../sample_molecules
for d in */ ; do
echo $d
	if [ -d "${d::-1}/prelim_energy" ] ; then
		cd ${d::-1}/prelim_energy
		for f in */ ; do
                     #   if [[ ${f::-1} == "run-0" || ${f::-1} == "run-6" || ${f::-1} == "run-7" ]] ; then
			cd $f
			echo ${d::-1} ${f::-1} $(grep "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):" output.out | tail -n 1 | awk '{print $(NF)}') >> $energy
			cd ..
		#	fi
		done	
		cd ../..
	fi
done

