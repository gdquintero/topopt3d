
from scipy.sparse import csr_matrix
from numpy import zeros

def global_stiff(struct, density, penal, row, col, inic, kel, pos):
    nodesx = int(struct.nnodesx)
    nnodes = int(struct.nodesNumber)
    nelemy = int(struct.nelemy)

    kk = zeros(inic[-1])

    #Primmera columna
    kk[pos[0, :]] +=  density[0]**penal * kel

    for i in range(1, nelemy - 1):
        kk[pos[i, :]] +=  density[i]**penal * kel 

    kk[pos[nelemy - 1 , :]] +=  density[nelemy - 1]**penal * kel


    #Columnas de la mitad y final
    for i in range(1, nodesx - 1):
        kk[pos[ i*nelemy , :]] +=  density[i*nelemy]**penal * kel

        for j in range(1, nelemy - 1):
            kk[pos[i*nelemy + j]] = density[i*nelemy + j]**penal * kel
        
        kk[pos[i*nelemy + nelemy - 1]] = density[i*nelemy + nelemy - 1]**penal * kel
        

    K = csr_matrix((kk, (row, col)), shape = (2*nnodes, 2*nnodes))
    return K




