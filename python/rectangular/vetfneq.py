import numpy as np

def vetfneq(struct):
    nodesNumber = int(struct["nodesNumber"])
    vectorForce = np.zeros(2*nodesNumber, dtype = np.double)
    structForcesNode = struct["forces"][0]["node"]
    structForcesFx = struct["forces"][0]["Fx"]
    structForcesFy = struct["forces"][0]["Fy"]
    structSupp = struct["supp"][0]
    structSuppNode = structSupp["node"]
    structSuppIx = structSupp["ix"]
    structSuppIy = structSupp["iy"]

    for i in range(len(structForcesNode)):
        vectorForce[2*structForcesNode[i] - 2] += structForcesFx[i]
        vectorForce[2*structForcesNode[i] - 1] += structForcesFy[i]

        for j in range(len(structSuppNode)):
            if structSuppIx[j] == 1:
                vectorForce[2*structSuppNode[j] - 2] == 0
            if structSuppIy[j] == 1:
                vectorForce[2*structSuppNode[j] - 1] == 0
    
    return vectorForce
                
