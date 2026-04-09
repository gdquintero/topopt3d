import numpy as np
import pandas as pd
from get_rows_cols import get_rows_cols
struct = pd.read_json("./python/triangularElements/struct3x3.json")

np.set_printoptions(threshold=np.inf)
(row, col, inic) = get_rows_cols(struct)
# print(inic)
def get_pos(struct, inic):
    nelem = int(struct.nelem)
    nelemy = int(struct.nelemy)
    nodesy = int(struct.nnodesy)
    nodesx = int(struct.nnodesx)

    pos = np.zeros((nelem, 36), dtype=int)
    # posOdd = np.zeros(inic[-1])
    # posEven = np.zeros(inic[-1])
    oind1 = np.array([1, 2, 5, 6, 7, 8], dtype=int)
    oind2 = np.array([1, 2, 3, 4, 5,6], dtype=int)
    oind3 = np.array([3, 4, 7, 8, 9, 10], dtype=int)
    oind4 = np.array([7, 8, 11, 12, 13, 14], dtype=int)

    eind1 = np.array([1, 2,7, 8, 3, 4], dtype = int)
    eind2 = np.array([3, 4, 9, 10, 5, 6], dtype = int)
    eind3 = np.array([1, 2, 5, 6, 3, 4], dtype= int)
    eind4 = np.array([5, 6, 11, 12, 7, 8], dtype=int)
    eind5 = np.array([7, 8, 13, 14, 9, 10], dtype=int)
    eind6 = np.array([5, 6, 9, 10, 7, 8], dtype=int)


    #Primera columna

    pos[0, :] = np.concatenate((inic[0] + oind1, inic[1] + oind1, 
                 inic[2*nodesy] + oind2, inic[2*nodesy + 1] + oind2, 
                 inic[2*nodesy + 2] + oind1, inic[2*nodesy + 3] + oind1))
    
    pos[1, :] = np.concatenate((inic[0] + eind1, inic[1] + eind1, 
                                inic[2*(nodesy + 2) - 2] + eind1, inic[2*(nodesy + 2) - 1] + eind1,
                                inic[2] + eind1, inic[3] + eind1))

    for i in range(2, nelemy - 3):
        
        pos[2*(i-1) + 1, :] = np.concatenate((inic[2*(i-1)] + eind2, inic[2*(i-1) + 1] + eind2, 
                                            inic[2*(i + nodesy)] + eind1, inic[2*(i + nodesy) + 1] + eind1, 
                                            inic[2*i] + eind1, inic[2*i + 1] + eind1))
        
        pos[2*(i - 1), :] = np.concatenate((inic[2*(i-1)] + oind3, inic[2*(i-1) + 1] + oind3, 
                                            inic[2*(i + nodesy - 1)] + oind3,  inic[2*(i + nodesy - 1) + 1] + oind3,
                                             inic[2*(i + nodesy)] + oind1, inic[2*(i + nodesy) + 1] + oind1 ))
   
   
    pos[nelemy - 2, :] = np.concatenate((inic[2*(nelemy - 4)] + oind3, inic[2*(nelemy - 4) + 1] + oind3,
                                        inic[2*(nelemy + nodesy - 4)] + oind3, inic[2*(nelemy + nodesy - 4) + 1] + oind3,
                                        inic[2*(nelemy + nodesy - 3)]+ oind1, inic[2*(nelemy + nodesy - 3) + 1] + oind1))
    
    pos[nelemy - 1, :] = np.concatenate((inic[2*(nelemy - 4)] + eind2, inic[2*(nelemy - 4) + 1] + eind2,
                                        inic[2*(nelemy + nodesy - 3)]+ eind1, inic[2*(nelemy + nodesy - 3) + 1] + eind1, 
                                        inic[2*(nelemy - 3)] + eind3, inic[2*(nelemy - 3) + 1] + eind3))
    
    #Columnas de enmedio
    
    for i in range(1, nodesx - 2):
        
        pos[i*nelemy, :] = np.concatenate((inic[2*(i*nodesy)] + oind3, inic[2*(i*nodesy) + 1] + oind3,
                                        inic[2*(i*nodesy + nodesy)] + oind2, inic[2*(i*nodesy + nodesy) + 1] + oind2, 
                                        inic[2*(i*nodesy + nodesy + 1)] + oind1, inic[2*(i*nodesy + nodesy + 1) + 1] + oind1))
        
        
        for j in range(2, nelemy - 3):

            pos[i*nelemy + 2*(j-1)] = np.concatenate((inic[2*(i*nodesy + (j-1))] + oind4, inic[2*(i*nodesy + (j-1)) + 1 ] + oind4,
                                        inic[2*(i*nodesy + nodesy + (j-1))] + oind3, inic[2*(i*nodesy + nodesy + (j-1)) + 1 ] + oind3, 
                                        inic[2*(i*nodesy + nodesy + 1 + (j-1))] + oind1, inic[2*(i*nodesy + nodesy + 1 + (j-1)) + 1] + oind1))
        
        
        pos[i*nelemy + nelemy - 2] = np.concatenate((inic[2*(i*nodesy + (nelemy - 5))] + oind4, inic[2*(i*nodesy + (nelemy - 5)) + 1 ] + oind4,
                                        inic[2*(i*nodesy + nodesy + (nelemy - 5))] + oind3, inic[2*(i*nodesy + nodesy + (nelemy - 5)) + 1 ] + oind3, 
                                        inic[2*(i*nodesy + nodesy + 1 + (nelemy - 5))] + oind1, inic[2*(i*nodesy + nodesy + 1 + (nelemy - 5)) + 1] + oind1))
    
    
    #Columna final
    pos[(nodesx - 2)*nelemy, :] = np.concatenate((inic[2*((nodesx - 2)*nodesy)] + oind3, inic[2*((nodesx - 2)*nodesy) + 1] + oind3,
                                        inic[2*((nodesx - 2)*nodesy + nodesy)] + oind2, inic[2*((nodesx - 2)*nodesy + nodesy) + 1] + oind2, 
                                        inic[2*((nodesx - 2)*nodesy + nodesy + 1)] + oind1, inic[2*((nodesx - 2)*nodesy + nodesy + 1) + 1] + oind1))
        
        
    for j in range(2, nelemy - 3):

        pos[(nodesx - 2)*nelemy + 2*(j-1)] = np.concatenate((inic[2*((nodesx - 2)*nodesy + (j-1))] + oind4, inic[2*((nodesx - 2)*nodesy + (j-1)) + 1 ] + oind4,
                                        inic[2*((nodesx - 2)*nodesy + nodesy + (j-1))] + oind3, inic[2*((nodesx - 2)*nodesy + nodesy + (j-1)) + 1 ] + oind3, 
                                        inic[2*((nodesx - 2)*nodesy + nodesy + 1 + (j-1))] + oind1, inic[2*((nodesx - 2)*nodesy + nodesy + 1 + (j-1)) + 1] + oind1))
        
        
    pos[(nodesx - 2)*nelemy + nelemy - 2] = np.concatenate((inic[2*((nodesx - 2)*nodesy + (nelemy - 5))] + oind4, inic[2*((nodesx - 2)*nodesy + (nelemy - 5)) + 1 ] + oind4,
                                        inic[2*((nodesx - 2)*nodesy + nodesy + (nelemy - 5))] + oind3, inic[2*((nodesx - 2)*nodesy + nodesy + (nelemy - 5)) + 1 ] + oind3, 
                                        inic[2*((nodesx - 2)*nodesy + nodesy + 1 + (nelemy - 5))] + oind1, inic[2*((nodesx - 2)*nodesy + nodesy + 1 + (nelemy - 5)) + 1] + oind1))
    # pos[]     
    
    return pos

print(get_pos(struct , inic))