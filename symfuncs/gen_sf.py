#!/usr/bin/env python3
import numpy as np

def g2(n,rc,elements,outfile):


    ms = np.arange(0,n)

    rsm = (rc/(n**(ms/n))).tolist()
    rsm.insert(0,0)

    lines = []

    for e1 in elements:
        for e2 in elements:

            for rs in rsm:
                for m in ms:
                    if rs == 0:
                        eta = ((n**(m/n))/rc)**2
                        lines.append("RADIAL " + str(e1) + " 2 " + str(e2) + " " + str(eta) + " 0 " + str(rc))
                    else:
                        eta = 1/(rsm[int(n-m)] - rsm[int(n-m-1)])**2
                        lines.append("RADIAL " + str(e1) + " 2 " + str(e2) + " " + str(eta) + " " + str(rs) + " " + str(rc))
    
    with open(outfile,'w') as file:
        for line in lines:
            file.write(line+"\n")


def g3(n,rc,elements,outfile):


    ms = np.arange(0,n)

    rsm = (rc/(n**(ms/n))).tolist()
    rsm.insert(0,0)

    zetas = [1,4,16]
    lambdas = [-1,1]

    lines = []

    for e1 in elements:
        for e2 in range(0,len(elements)):
            for e3 in range(e2,len(elements)):
                for lam in lambdas:
                    for zeta in zetas:
                        for rs in rsm:
                            for m in ms:
                                if rs == 0:
                                    eta = ((n**(m/n))/rc)**2
                                    lines.append("ANGULAR_NARROW " + str(e1) + " 3 " + str(elements[e2]) + " " + str(elements[e3]) + " " + str(eta) + " " + str(lam) + " " + str(zeta) +  " 0 " + str(rc))
                                else:
                                    eta = 1/(rsm[int(n-m)] - rsm[int(n-m-1)])**2
                                    lines.append("ANGULAR_NARROW " + str(e1) + " 3 " + str(elements[e2]) + " " + str(elements[e3]) + " " + str(eta) + " " + str(lam) + " " + str(zeta) + " " + str(rs) + " "+ str(rc))
        
    with open(outfile,'a') as file:
        for line in lines:
            file.write(line+"\n")



def g9(n,rc,elements,outfile):


    ms = np.arange(0,n)

    rsm = (rc/(n**(ms/n))).tolist()
    rsm.insert(0,0)

    zetas = [1,4,16]
    lambdas = [-1,1]

    lines = []

    for e1 in elements:
        for e2 in range(0,len(elements)):
            for e3 in range(e2,len(elements)):
                for lam in lambdas:
                    for zeta in zetas:
                        for rs in rsm:
                            for m in ms:
                                if rs == 0:
                                    eta = ((n**(m/n))/rc)*2
                                    lines.append("ANGULAR_WIDE " + str(e1) + " 9 " + str(elements[e2]) + " " + str(elements[e3]) + " " + str(eta) + " " + str(lam) + " " + str(zeta) +  " 0 " + str(rc))
                                else:
                                    eta = 1/(rsm[int(n-m)] - rsm[int(n-m-1)])**2
                                    lines.append("ANGULAR_WIDE " + str(e1) + " 9 " + str(elements[e2]) + " " + str(elements[e3]) + " " + str(eta) + " " + str(lam) + " " + str(zeta) + " " + str(rs) + " "+ str(rc))
        
    with open(outfile,'a') as file:
        for line in lines:
            file.write(line+"\n")


##Alter as required
elements=["H","B","C","N","O","F","S","Cl","Ln"]
outfile="sfout.txt"

n = 2
rc = 6

g2(n,rc,elements,outfile)
g3(n,rc,elements,outfile)
g9(n,rc,elements,outfile)
