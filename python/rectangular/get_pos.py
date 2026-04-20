import numpy as np

def get_pos(struct, inic):
    nelem = int(struct.nelem)
    nelemy = int(struct.nelemy)
    nodesy = int(struct.nnodesy)
    nodesx = int(struct.nnodesx)

    pos = np.zeros((nelem, 64), dtype=int)
    # posOdd = np.zeros(inic[-1])
    # posEven = np.zeros(inic[-1])
    ind1 = np.array([1, 2, 5, 6, 7, 8, 3, 4], dtype = int)
    ind2 = np.array([1, 2, 7, 8, 9, 10, 3, 4], dtype = int)
    ind3 = np.array([3, 4, 9, 10, 11, 12, 5, 6], dtype= int)
    ind4 = np.array([5, 6, 9, 10, 11, 12, 7, 8], dtype=int)
    ind5 = np.array([7, 8, 13, 14, 15, 16, 9, 10], dtype=int)
    ind6 = np.array([9, 10, 15, 16, 17, 18, 11, 12], dtype=int)


    #Primera columna

    pos[0, :] = np.concatenate((inic[0] + ind1, inic[1] + ind1, 
                 inic[2*nodesy] + ind1, inic[2*nodesy + 1] + ind1, 
                 inic[2*nodesy + 2] + ind2, inic[2*nodesy + 3] + ind2,
                 inic[2] + ind2, inic[3] + ind2))
    

    for i in range(2, nelemy):
        
        pos[i - 1, :] = np.concatenate((inic[2*(i-1)] + ind3, inic[2*(i-1) + 1] + ind3, 
                                            inic[2*(i + nodesy - 1)] + ind3,  inic[2*(i + nodesy - 1) + 1] + ind3,
                                            inic[2*(i + nodesy)] + ind2, inic[2*(i + nodesy) + 1] + ind2, 
                                            inic[2*i] + ind2, inic[2*i + 1] + ind2
                                            ))

   
       
    pos[nelemy - 1, :] = np.concatenate((inic[2*(nodesy - 2)] + ind3, inic[2*(nodesy - 2) + 1] + ind3,
                                        inic[2*((nodesy - 2) + nodesy)] + ind3, inic[2*((nodesy - 2) + nodesy) + 1] + ind3,
                                        inic[2*((nodesy - 2) + nodesy + 1)]+ ind1, inic[2*((nodesy - 2) + nodesy  + 1) + 1] + ind1, 
                                        inic[2*(nodesy - 1) ] + ind1, inic[2*(nodesy - 1) + 1] + ind1))
    
    #Columnas de enmedio y final
    
    for i in range(1, nodesx - 1):
                
        pos[i*nelemy, : ] = np.concatenate((inic[2*(i*nodesy)] + ind4, inic[2*(i*nodesy) + 1] + ind4, 
                                                inic[2*(i*nodesy + nodesy)] + ind1, inic[2*(i*nodesy + nodesy) + 1] + ind1, 
                                                 inic[2*(i*nodesy + nodesy + 1)] + ind2, inic[2*(i*nodesy + nodesy + 1) + 1] + ind2,
                                                 inic[2*(i*nodesy + 1)] + ind5, inic[2*(i*nodesy + 1) + 1] + ind5))
        
        for j in range(2, nelemy):

            pos[i*nelemy + j - 1] = np.concatenate((inic[2*(i*nodesy + (j-1))] + ind6, inic[2*(i*nodesy + (j-1)) + 1 ] + ind6,
                                        inic[2*(i*nodesy + nodesy + (j-1))] + ind3, inic[2*(i*nodesy + nodesy + (j-1)) + 1 ] + ind3, 
                                        inic[2*(i*nodesy + nodesy + 1 + (j-1))] + ind2, inic[2*(i*nodesy + nodesy + 1 + (j-1)) + 1] + ind2, 
                                        inic[2*(i*nodesy + (j-1) + 1)] + ind5, inic[2*(i*nodesy + (j-1)  + 1) + 1 ] + ind5))
        
        
        pos[i*nelemy + nelemy - 1] = np.concatenate((inic[2*(i*nodesy + (nodesy - 2))] + ind6, inic[2*(i*nodesy + (nodesy - 2)) + 1 ] + ind6,
                                        inic[2*(i*nodesy + nodesy + (nodesy - 2))] + ind3, inic[2*(i*nodesy + nodesy + (nodesy - 2)) + 1 ] + ind3, 
                                        inic[2*(i*nodesy + nodesy + 1 + (nodesy - 2))] + ind1, inic[2*(i*nodesy + nodesy + 1 + (nodesy - 2)) + 1] + ind1,
                                        inic[2*(i*nodesy + (nodesy - 2) + 1)] + ind4, inic[2*(i*nodesy + (nodesy - 2) + 1) + 1 ] + ind4))
        
    
    return pos

# print(get_pos(struct , inic))