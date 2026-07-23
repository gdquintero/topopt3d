from numpy import zeros
from numpy import floor

def linel(struct):
    ny = int(struct["nnodesy"])
    nelem = int(struct["nelem"])
    k = ny - 1

    lin = zeros((8, nelem), dtype=int)

    for i in range(1, nelem + 1):
         node = (floor((i-1)/k))*ny+((i-1) % k)+1
         lin[1, i - 1] =  2*node
         lin[0, i - 1] = lin[1, i - 1] - 1

         lin[3, i  - 1] = lin[1, i - 1] + 2*ny
         lin[2, i - 1] = lin[3, i - 1] - 1

         lin[5, i - 1] = lin[3, i - 1] + 2
         lin[4, i - 1] = lin[5, i - 1] - 1

         lin[7, i - 1] = lin[1, i - 1] + 2
         lin[6, i - 1] = lin[7, i - 1] - 1
         
    return lin

