from linel import linel
from numpy import zeros
def grad_compliance(struct, u, penal, density, keven, kodd):
    df0dx = zeros(int(struct.nelem))
    lin = linel(struct)
    lin = lin - 1

    for i in range(int(struct.nelem/2)):
        df0dx[2*i] = -penal*(density[2*i]**(penal - 1)) * u[lin[:, 2*i]] @ kodd @ u[lin[:, 2*i]]  
        df0dx[2*i + 1] = -penal*(density[2*i + 1]**(penal - 1)) * u[lin[:, 2*i + 1]] @ keven @ u[lin[:, 2*i + 1]]  

    return df0dx