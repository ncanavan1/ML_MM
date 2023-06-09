#!/bin/bash

rm -r fullSMILES
mkdir fullSMILES
echo STARTING 4 4 MOLECULES
./gen_4_4.sh 4_connections/ 4_connections/
echo STARTING 4 2 2 MOLECULES
./gen_4_2x2.sh 4_connections/ 2_connections/ 2_connections/
echo STARTING 4 1 1 1 1 MOLECULES
./gen_4_4x1.sh 4_connections/ 1_connections/ 1_connections/ 1_connections/ 1_connections/
echo STARTING 4 2 1 1 MOLECULES
./gen_4_2_1_1.sh 4_connections/ 2_connections/ 1_connections/ 1_connections/
echo STARTING 2 2 2 2 MOLECULES
./gen_2x4.sh 2_connections/ 2_connections/ 2_connections/ 2_connections/
echo STARTING 2 2 2 1 1 MOLECULES
./gen_2x3_2x1.sh 2_connections/ 2_connections/ 2_connections/ 1_connections/ 1_connections/
echo STARTING 2 2 1 1 1 1 MOLECULES
./gen_2x2_4x1.sh 2_connections/ 2_connections/ 1_connections/ 1_connections/ 1_connections/ 1_connections/
echo ALL DONE
