#!/usr/bin/env python3

import numpy as np
import sys
import csv
import pandas as pd

csvfile = sys.argv[1] ##overfull evaluated symfunctions
energycsv = sys.argv[2] ##energy for molecules
d = int(sys.argv[3]) ##reduction number
outputfile = sys.argv[4] ##output file name


x = pd.read_csv(csvfile,header=None)
x=x.values
names=x[:,0]
x=x[:,1:]

energydf = pd.read_csv(energycsv,header=None)
energydf = energydf.values



def do_fps(x, d=0,initial=-1):
    # Code from Giulio Imbalzano

    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    if (initial == -1):
        iy[0] = np.random.randint(0,n)
    else:
        iy[0] = initial
    # Faster evaluation of Euclidean distance
    # Here we fill the n2 array in this way because it halves the memory cost of this routine
    n2 = np.array([np.sum(x[i] * np.conj([x[i]])) for i in range(len(x))])
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))

    for i in range(1,d):
        print("Doing ",i," of ",d," dist = ",max(dl))
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

y = do_fps(x.transpose(),d)
x_pruned = np.zeros([x.shape[0],len(y)])

with open("reduced_sf_list.txt", "w") as f:
    for ys in y:
         f.write(str(ys) + "\n")

j=0
for i in range(0,x.shape[1]):
    if i in y:
       x_pruned[:,j] = x[:,i]
       j = j+1


with open(outputfile, "w") as csvfile:
    writer=csv.writer(csvfile)
    header = ["Molecule"]
    for i in range(0,d):
        header.append("SF" + str(i))
    header.append("Energy")
    writer.writerow(header)
    for row in range(0,x_pruned.shape[0]):
        info=[names[row]]
        info = info + x_pruned[row].tolist() 	
        for m in range(0,energydf.shape[0]):
            m_curr = energydf[m,0]
            if m_curr == names[row]:
                info = info + [energydf[m,1]]
        writer.writerow(info)
    csvfile.flush()
