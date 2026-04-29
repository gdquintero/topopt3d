
from scipy.sparse import csr_matrix
from numpy import zeros

def global_stiff(struct, density, penal, row, col, inic, kelE, kelO, pos):
    nodesx = int(struct.nnodesx)
    nnodes = int(struct.nodesNumber)
    nelemy = int(struct.nelemy)

    kk = zeros(inic[-1])

    #Primmera columna
    kk[pos[0, :]] +=  density[0]**penal * kelO
    kk[pos[1, :]] +=  density[1]**penal * kelE
    for i in range(1, int(nelemy/2 - 1)):
        kk[pos[2*i, :]] +=  density[2*i]**penal * kelO
        kk[pos[2*i + 1, :]] +=  density[2*i + 1]**penal * kelE

    kk[pos[nelemy - 2, :]] +=  density[nelemy - 2]**penal * kelO
    kk[pos[nelemy - 1 , :]] +=  density[nelemy - 1]**penal * kelE

    #Columnas de la mitad y final
    for i in range(1, nodesx - 1):
        kk[pos[ i*nelemy , :]] +=  density[i*nelemy]**penal * kelO
        kk[pos[ i*nelemy + 1 , :]] +=  density[i*nelemy + 1]**penal * kelE

        for j in range(1, int(nelemy/2 - 1)):
            kk[pos[i*nelemy + 2*j]] += density[i*nelemy + 2*j]**penal * kelO
            kk[pos[i*nelemy + 2*j  + 1]] += density[i*nelemy + 2*j + 1]**penal * kelE
        
        kk[pos[i*nelemy + nelemy - 2]] += density[i*nelemy + nelemy - 2]**penal * kelO
        kk[pos[i*nelemy + nelemy - 1]] += density[i*nelemy + nelemy - 1]**penal * kelE
        
    K = csr_matrix((kk, (row, col)), shape = (2*nnodes, 2*nnodes))
    return K




