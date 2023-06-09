#!/bin/bash

export num=${1}

rm -r ../molecules
mkdir ../molecules

ls fullSMILES/ | sort -R | tail -$num | while read file ; do
		cd ../molecules
		mkdir ${file::-4}
		cd ${file::-4}
		cp ../../smiles_gen/fullSMILES/${file} .
		obabel -ismi ${file} -ocan > ${file::-4}_can.smi
		molconvert -3 PDB:H ${file::-4}_can.smi -o ${file::-4}.pdb
		obabel -ipdb ${file::-4}.pdb -oxyz > ${file::-4}.xyz
		cd ../../smiles_gen	
done
echo ALL DONE :\)
