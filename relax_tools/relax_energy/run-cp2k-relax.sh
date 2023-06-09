#!/bin/bash
export inputframe=${1}
export maxmultiplicity=${2}

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
sed -i "s/ABC 25.0 25.0 25.0/${cell}/g" geo-opt-relax.inp

let "mm1=$maxmultiplicity - 1"
for multiplicity in $(seq ${maxmultiplicity} -1 0);do
  if [ "$multiplicity" == "$maxmultiplicity" ] || [ "$multiplicity" == "0" ] || [ $multiplicity == $mm1 ] ; then
      mkdir -p run-${multiplicity}
      cd run-${multiplicity}
      cat ../geo-opt-relax.inp | sed "s/MLTP/${multiplicity}/" > geo-opt-relax.inp 
      source ~/source/cp2k/tools/toolchain/install/setup
      echo $inputframe run-${multiplicity}
      mpirun -np 121 ~/source/cp2k/exe/local/cp2k.popt geo-opt-relax.inp &> output.out
      cd ../
  fi
done
