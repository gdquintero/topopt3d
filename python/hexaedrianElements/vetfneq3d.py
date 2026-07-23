import numpy as np


def vetfneq3d(struct):
    forcesVector = np.zeros(3*struct.nodesNumber)
    for i in range(len(struct.forces[0]["node"])):
        forcesVector[3*struct.forces[0]["node"][i] - 3] += struct.forces[0]["Fx"][i]
        forcesVector[3*struct.forces[0]["node"][i] - 2] += struct.forces[0]["Fy"][i]
        forcesVector[3*struct.forces[0]["node"][i] - 1] += struct.forces[0]["Fz"][i]

        for j in range(len(struct.supp[0]["node"])):
            if struct.supp[0]["ix"][j] == 1:
                forcesVector[3*struct.supp[0]["node"][j] - 3] = 0
            
            if struct.supp[0]["iy"][j] == 1: 
                forcesVector[3*struct.supp[0]["node"][j] - 2] = 0
            
            if struct.supp[0]["iz"][j] == 1: 
                forcesVector[3*struct.supp[0]["node"][j] - 1] = 0
    return forcesVector
                

