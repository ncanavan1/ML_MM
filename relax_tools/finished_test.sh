#!/bin/bash

unfinishedfile="unfinished.txt"
rm $unfinishedfile
touch $unfinishedfile
cd ../molecules
for d in */ ; do
	cd $d
	if [ -d "geometry_optimisation" ] ; then
		cd geometry_optimisation
		for f in */ ; do
			cd $f
			if [[ $(cat output.out | tail -1) ==  *"DUE TO TIME LIMIT"* ]] ; then
				echo "$d $f" >> $unfinishedfile
			fi
			cd ..
		done	
	        cd ..
	fi
        cd ..
done
