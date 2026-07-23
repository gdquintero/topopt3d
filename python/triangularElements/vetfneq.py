import numpy as np

def vetfneq(struct):
    vectorForce = np.zeros(2*struct.nodesNumber)
    for i in range(len(struct.forces[0]["node"])):
        vectorForce[2*struct.forces[0]["node"][i] - 2] += struct.forces[0]["Fx"][i]
        vectorForce[2*struct.forces[0]["node"][i] - 1] += struct.forces[0]["Fy"][i]

        for j in range(len(struct.supp[0]["node"])):
            if struct.supp[0]["ix"][j] == 1:
                vectorForce[2*struct.supp[0]["node"][j] - 2] == 0
            if struct.supp[0]["iy"][j] == 1:
                vectorForce[2*struct.supp[0]["node"][j] - 1] == 0
    
    return vectorForce
                
