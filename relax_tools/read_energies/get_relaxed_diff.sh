#!/bin/bash

optdir="geometry_optimisation"
energy="$PWD/../../relaxed_energy.txt"

rm $energy
touch $energy

cd ../../sample_molecules

for d in */ ; do
        if [ -d $d/$optdir ] ; then
        cd $d/$optdir
        for f in */ ; do
            cd $f
            ##if finished check
            if grep -q "The number of warnings for this run is" output.out ; then
                if grep -q "MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED" output.out ; then
			echo ${d::-1} ${f::-1} Max Steps >> $energy
		else
                	echo ${d::-1} ${f::-1} $(grep "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):" output.out | tail -n 1 | awk '{print $(NF)}') >> $energy
           	fi
	    else
                echo ${d::-1} ${f::-1} Unfinished/Invalid MTLP >> $energy
            fi
            cd ..
        done
        cd ../..
        fi
done
