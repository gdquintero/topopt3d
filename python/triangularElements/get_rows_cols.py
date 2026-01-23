from numpy import zeros
from numpy import array
import pandas as pd

struct = pd.read_json("./python/triangularElements/struct3x3.json")


def get_rows_cols(struct):
    ny = int(struct.nnodesy)
    nx = int(struct.nnodesx)
    ncol = int(2*struct.nodesNumber)
    inic = zeros(ncol + 1, dtype= int)
    nelmat = 56 + (ny - 2)*40 + (nx - 2)*40 + 28*(ny - 2)*(nx - 2)
    row = zeros(nelmat)
    col = zeros(nelmat)
    ant = array([-2*ny-3, -2*ny-2,-2*ny-1, -2*ny, -2*ny+1, -2*ny+2])
    atu = array([-3, -2, -1, 0, 1, 2])
    pro = array([2*ny-3, 2*ny-2, 2*ny-1, 2*ny, 2*ny+1, 2*ny+2])
    
    #Primera fila de elementos 
    nel = 8 
    inic[1] = inic[0] + nel
    inic[2] = inic[1] + nel
    col[inic[0] : inic[1]] = 1 
    col[inic[1]: inic[2]] = 2

    for i in range(1, ny -1 ):
        nel = 10
        inic[2*i + 1] = inic[2*i] + nel
        inic[2*(i+1)] = inic[2*i + 1] + nel
        col[inic[2*i] : inic[2*i + 1]] = 2*i + 1
        col[inic[2*i + 1] : inic[2*(i+1)] ] = 2*i + 2

    nel = 6
    inic[2*( ny - 1) + 1]  = inic[2*(ny - 1)] + nel 
    inic[2*(ny - 1) + 2] = inic[2*(ny - 1) + 1] + nel
    col[inic[2*(ny - 1)] : inic[2*( ny - 1) + 1] ] = 2*(ny - 1) + 1
    col[inic[2*( ny - 1) + 1] : inic[2*( ny - 1) + 2] ] = 2*(ny - 1) + 2
    
    #linea de en medio 
    for i in range(1, nx - 1):
        nel = 10
        inic[2*(i*ny + 1) - 1] = inic[2*(i*ny + 1) - 2] + nel
        inic[2*(i*ny + 1)] = inic[2*(i*ny + 1) - 1] + nel
        
        col[inic[2*(i*ny + 1) - 2] : inic[2*(i*ny + 1) - 1] ] = 2*(i*ny + 1) - 1
        col[inic[2*(i*ny + 1) - 1] : inic[2*(i*ny + 1)] ] = 2*(i*ny + 1) 
        
        for j in range(1, ny - 1):
            nel = 14
            inic[2*(i*ny + 1 + j) - 1] = inic[2*(i*ny + 1 + j) - 2] + nel
            inic[2*(i*ny + 1 + j)] = inic[2*(i*ny + 1 + j) - 1] + nel
            
            col[inic[2*(i*ny + 1 + j) - 2] : inic[2*(i*ny + 1 + j) - 1]] = 2*(i*ny + 1 + j) - 1
            col[inic[2*(i*ny + 1 + j) - 1] : inic[2*(i*ny + 1 + j) ]] = 2*(i*ny + 1 + j) 
        
        nel = 10
        inic[2*ny*(i + 1) - 1] = inic[2*ny*(i + 1) - 2] + nel
        inic[2*ny*(i + 1)] = inic[2*ny*(i + 1) - 1] + nel
        col[inic[2*ny*(i + 1) - 2] : inic[2*ny*(i + 1) - 1] ] = 2*ny*(i + 1) - 1 
        col[inic[2*ny*(i + 1) - 1] : inic[2*ny*(i + 1)] ] = 2*ny*(i + 1) 
        
    #Linea finall
    nel = 6
    inic[2*(ny*(nx - 1) + 1) - 1] = inic[2*(ny*(nx - 1) + 1) - 2] + nel
    inic[2*(ny*(nx - 1) + 1)] =  inic[2*(ny*(nx - 1) + 1) - 1] + nel

    col[inic[2*(ny*(nx - 1) + 1) - 2] : inic[2*(ny*(nx - 1) + 1) - 1] ] = 2*(ny*(nx - 1) + 1) - 1
    col[inic[2*(ny*(nx - 1) + 1) - 1] : inic[2*(ny*(nx - 1) + 1)] ] = 2*(ny*(nx - 1) + 1) 

    for i in range(1, ny - 1):
        nel = 10
        inic[2*(ny*(nx - 1) + 1 + i) - 1] = inic[2*(ny*(nx - 1) + 1 + i) - 2] + nel 
        inic[2*(ny*(nx - 1) + 1 + i)] = inic[2*(ny*(nx - 1) + 1 + i) - 1] + nel
        col[inic[2*(ny*(nx - 1) + 1 + i) - 2] : inic[2*(ny*(nx - 1) + 1 + i) - 1] ] = 2*(ny*(nx - 1) + 1 + i) - 1 
        col[inic[2*(ny*(nx - 1) + 1 + i) - 1] : inic[2*(ny*(nx - 1) + 1 + i)] ] = 2*(ny*(nx - 1) + 1 + i)

    nel = 8
    inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] = inic[2*(ny*(nx - 1) + 1 + ny - 1) - 2] + nel 
    inic[2*(ny*(nx - 1) + 1 + ny - 1)] = inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] + nel
    col[inic[2*(ny*(nx - 1) + 1 + ny - 1) - 2] : inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] ] = 2*(ny*(nx - 1) + 1 + ny - 1) - 1 
    col[inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] : inic[2*(ny*(nx - 1) + 1 + ny - 1)] ] = 2*(ny*(nx - 1) + 1 + ny - 1)
    print(inic, col.shape)
    return col

print(get_rows_cols(struct))