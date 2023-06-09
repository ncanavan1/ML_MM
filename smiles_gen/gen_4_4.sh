#!/bin/bash

full="fullSMILES"
lnlist="ln_list"
export list1=${1} #directory for 4 connection ligands
export list2=${2} #directory for 4 connection ligands

i=0
for l1 in ${list1}*.smi ; do
	j=0
	for l2 in ${list2}*.smi ; do
		fn1=$(basename -- "$l1")
		fn2=$(basename -- "$l2")
                while read lanth ; do
		    fname=${lanth}_${fn1::-4}_${fn2::-4}.smi
		    cat templates/2_Lig_template | sed "s/L1/$(cat $l1)/g;s/L2/$(cat $l2)/g;s/Ln/${lanth}/g" > ${full}/$fname
                done < $lnlist
	((j++))
	done
((i++))
done

