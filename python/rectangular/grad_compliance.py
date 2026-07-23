from numpy import zeros
# from numpy import floor
import pandas as pd
from linel import linel
from numpy import double

def grad_compliance(struct, u,  penal, density, kel):
    nelem = int(struct["nelem"])
    df0dx = zeros(nelem, dtype=double)

    # ny = int(struct.nnodesy)
    # k = ny - 1

    lin = linel(struct)
    lin = lin - 1

    for i in range(nelem):
        df0dx[i] = -penal*(density[i]**(penal - 1)) * u[lin[:, i]] @ kel @ u[lin[:, i]]  

    return df0dx

   