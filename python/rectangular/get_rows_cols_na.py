from numpy import zeros
from numpy import array
from numpy import concatenate

def get_rows_cols(struct):
    ny = int(struct.nnodesy)
    nx = int(struct.nnodesx)
    ncol = int(2*struct.nodesNumber)
    inic = zeros(ncol + 1, dtype= int)
    nelmat = 64 + (ny - 2)*48 + (nx - 2)*48 + 36*(ny - 2)*(nx - 2)
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

    row[inic[0] : inic[1]] = 2 + concatenate((atu[2:], pro[2:]))
    row[inic[1] : inic[2]] = 2 + concatenate((atu[2:], pro[2:]))

    
    for i in range(1, ny -1 ):
        nel = 12
        inic[2*i + 1] = inic[2*i] + nel
        inic[2*(i+1)] = inic[2*i + 1] + nel
        
        col[inic[2*i] : inic[2*i + 1]] = 2*i + 1
        col[inic[2*i + 1] : inic[2*(i+1)] ] = 2*i + 2
        
        row[inic[2*i] : inic[2*i + 1]] = 2*i + 2 + concatenate((atu, pro))
        row[inic[2*i + 1] : inic[2*i + 2]] = 2*i + 2 + concatenate((atu, pro))

    nel = 8
    inic[2*( ny - 1) + 1]  = inic[2*(ny - 1)] + nel 
    inic[2*(ny - 1) + 2] = inic[2*(ny - 1) + 1] + nel
    
    col[inic[2*(ny - 1)] : inic[2*( ny - 1) + 1] ] = 2*(ny - 1) + 1
    col[inic[2*( ny - 1) + 1] : inic[2*( ny - 1) + 2] ] = 2*(ny - 1) + 2
    
    row[inic[2*(ny - 1)] : inic[2*( ny - 1) + 1]] = 2*(ny - 1) + 2 + concatenate((atu[0:4], pro[0:4]))
    row[inic[2*(ny - 1) + 1] : inic[2*( ny - 1) + 2]] = 2*(ny - 1) + 2 + concatenate((atu[0:4], pro[0:4]))
    

    #linea de en medio 
    for i in range(1, nx - 1):
        nel = 12
        inic[2*(i*ny + 1) - 1] = inic[2*(i*ny + 1) - 2] + nel
        inic[2*(i*ny + 1)] = inic[2*(i*ny + 1) - 1] + nel
        
        col[inic[2*(i*ny + 1) - 2] : inic[2*(i*ny + 1) - 1] ] = 2*(i*ny + 1) - 1
        col[inic[2*(i*ny + 1) - 1] : inic[2*(i*ny + 1)] ] = 2*(i*ny + 1) 
        
        row[inic[2*(i*ny + 1) - 2] : inic[2*(i*ny + 1) - 1] ] = 2*(i*ny + 1) + concatenate((ant[2:], atu[2:], pro[2:]))  
        row[inic[2*(i*ny + 1) - 1] : inic[2*(i*ny + 1)] ] = 2*(i*ny + 1) + concatenate((ant[2:], atu[2:], pro[2:]))
        

        for j in range(1, ny - 1):
            nel = 18
            inic[2*(i*ny + 1 + j) - 1] = inic[2*(i*ny + 1 + j) - 2] + nel
            inic[2*(i*ny + 1 + j)] = inic[2*(i*ny + 1 + j) - 1] + nel
            
            col[inic[2*(i*ny + 1 + j) - 2] : inic[2*(i*ny + 1 + j) - 1]] = 2*(i*ny + 1 + j) - 1
            col[inic[2*(i*ny + 1 + j) - 1] : inic[2*(i*ny + 1 + j) ]] = 2*(i*ny + 1 + j)
            
            row[inic[2*(i*ny + 1 + j) - 2] : inic[2*(i*ny + 1 + j) - 1]] = 2*(i*ny + 1 + j) + concatenate((ant, atu, pro))
            row[inic[2*(i*ny + 1 + j) - 1] : inic[2*(i*ny + 1 + j) ]] = 2*(i*ny + 1 + j) + concatenate((ant, atu, pro))

              
        nel = 12
        inic[2*ny*(i + 1) - 1] = inic[2*ny*(i + 1) - 2] + nel
        inic[2*ny*(i + 1)] = inic[2*ny*(i + 1) - 1] + nel
        col[inic[2*ny*(i + 1) - 2] : inic[2*ny*(i + 1) - 1] ] = 2*ny*(i + 1) - 1 
        col[inic[2*ny*(i + 1) - 1] : inic[2*ny*(i + 1)] ] = 2*ny*(i + 1)

        row[inic[2*ny*(i + 1) - 2] : inic[2*ny*(i + 1) - 1] ] = 2*ny*(i + 1) + concatenate((ant[0:4], atu[0:4], pro[0:4])) 
        row[inic[2*ny*(i + 1) - 1] : inic[2*ny*(i + 1)] ] = 2*ny*(i + 1) + concatenate((ant[0:4], atu[0:4], pro[0:4]))
        
    
    #Linea finall
    nel = 8
    inic[2*(ny*(nx - 1) + 1) - 1] = inic[2*(ny*(nx - 1) + 1) - 2] + nel
    inic[2*(ny*(nx - 1) + 1)] =  inic[2*(ny*(nx - 1) + 1) - 1] + nel

    col[inic[2*(ny*(nx - 1) + 1) - 2] : inic[2*(ny*(nx - 1) + 1) - 1] ] = 2*(ny*(nx - 1) + 1) - 1
    col[inic[2*(ny*(nx - 1) + 1) - 1] : inic[2*(ny*(nx - 1) + 1)] ] = 2*(ny*(nx - 1) + 1) 
    
    row[inic[2*(ny*(nx - 1) + 1) - 2] : inic[2*(ny*(nx - 1) + 1) - 1] ] = 2*(ny*(nx - 1) + 1) + concatenate((ant[2:], atu[2:]))
    row[inic[2*(ny*(nx - 1) + 1) - 1] : inic[2*(ny*(nx - 1) + 1)] ] = 2*(ny*(nx - 1) + 1) + + concatenate((ant[2:], atu[2:]))


    for i in range(1, ny - 1):
        nel = 12
        inic[2*(ny*(nx - 1) + 1 + i) - 1] = inic[2*(ny*(nx - 1) + 1 + i) - 2] + nel 
        inic[2*(ny*(nx - 1) + 1 + i)] = inic[2*(ny*(nx - 1) + 1 + i) - 1] + nel

        col[inic[2*(ny*(nx - 1) + 1 + i) - 2] : inic[2*(ny*(nx - 1) + 1 + i) - 1] ] = 2*(ny*(nx - 1) + 1 + i) - 1 
        col[inic[2*(ny*(nx - 1) + 1 + i) - 1] : inic[2*(ny*(nx - 1) + 1 + i)] ] = 2*(ny*(nx - 1) + 1 + i)

        row[inic[2*(ny*(nx - 1) + 1 + i) - 2] : inic[2*(ny*(nx - 1) + 1 + i) - 1] ] = 2*(ny*(nx - 1) + 1 + i) + concatenate((ant, atu)) 
        row[inic[2*(ny*(nx - 1) + 1 + i) - 1] : inic[2*(ny*(nx - 1) + 1 + i)] ] = 2*(ny*(nx - 1) + 1 + i) + concatenate((ant, atu))



    nel = 8
    inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] = inic[2*(ny*(nx - 1) + 1 + ny - 1) - 2] + nel 
    inic[2*(ny*(nx - 1) + 1 + ny - 1)] = inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] + nel

    col[inic[2*(ny*(nx - 1) + 1 + ny - 1) - 2] : inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] ] = 2*(ny*(nx - 1) + 1 + ny - 1) - 1 
    col[inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] : inic[2*(ny*(nx - 1) + 1 + ny - 1)] ] = 2*(ny*(nx - 1) + 1 + ny - 1)

    row[inic[2*(ny*(nx - 1) + 1 + ny - 1) - 2] : inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] ] = 2*(ny*(nx - 1) + 1 + ny - 1) + concatenate((ant[0:4], atu[0:4])) 
    row[inic[2*(ny*(nx - 1) + 1 + ny - 1) - 1] : inic[2*(ny*(nx - 1) + 1 + ny - 1)] ] = 2*(ny*(nx - 1) + 1 + ny - 1) + concatenate((ant[0:4], atu[0:4]))
    
    
    return (row, col, inic)
