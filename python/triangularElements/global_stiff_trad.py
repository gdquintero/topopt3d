from linel import linel
from numpy import zeros
def global_stiff_trad(struct, kelEven, kelOdd, x, p):
    lin = linel(struct)
    lin = lin - 1
    K = zeros((2*int(struct["nodesNumber"]), 2*int(struct["nodesNumber"])))
    for k in range(int(struct["nelem"]/2)):
        for i in range(6):
            for j in range(6):
                K[lin[i, 2*k], lin[j, 2*k]] += (x[2*k]**p)*kelOdd[i, j]
                K[lin[i, 2*k + 1], lin[j, 2*k + 1]] += (x[2*k + 1]**p)*kelEven[i, j]
    return K