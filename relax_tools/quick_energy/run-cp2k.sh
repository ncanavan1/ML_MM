#!/bin/bash
export inputframe=${1}
export maxmultiplicity=${2}

##adjust cell size for molecule
./calculate_cell.py ../frame.xyz 
line_no=0
while read -r line; do
	if ((line_no == 0)) ; then
		xr=$line
	fi
        if ((line_no == 1)) ; then
                yr=$line
        fi
        if ((line_no == 2)) ; then
                zr=$line
        fi
	((line_no++))
done < tmp.txt
cell="ABC ${xr} ${yr} ${zr}"
sed -i "s/ABC 25.0 25.0 25.0/${cell}/g" geo-opt.inp



for multiplicity in $(seq ${maxmultiplicity} -1 0);do
  mkdir -p run-${multiplicity}
  cd run-${multiplicity}
  cat ../geo-opt.inp | sed "s/MLTP/${multiplicity}/" > geo-opt.inp 
  source ~/source/cp2k/tools/toolchain/install/setup
  mpirun -np 121 ~/source/cp2k/exe/local/cp2k.popt geo-opt.inp &> output.out
  cd ../
done

