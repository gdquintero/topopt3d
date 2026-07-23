# import pandas as pd
# from weight import weight
from numpy import floor, zeros, sqrt
# struct = pd.read_json("./python/rectangular/struct5x4.json")

def get_neighborhood(struct, rmin):
    rminf = int(floor(rmin))
    nelem = int(struct["nelem"])
    nrow = (2*rminf + 1)**2

    neighbsEl = zeros((nrow, nelem), dtype=int)
    numNei = zeros(nelem, dtype=int)
    nelemx = int(struct["nelemx"])
    nelemy = int(struct["nelemy"])
    hx = struct["b"]/nelemx
    hy = struct["h"]/nelemy

    distnei = zeros((nrow, nelem))
    for i in range(1, nelemx + 1):
        for j in range(1, nelemy + 1):
            m1 = (i - 1)*nelemy + j - 1
            ind = 0
            kmin = max(i-rminf,1)
            kmax = min(i+rminf,nelemx)
            lmin = max(j-rminf,1)
            lmax = min(j+rminf,nelemy)
            for k in range(kmin, kmax + 1):
                for l in range(lmin, lmax + 1):
                    m2 = (k - 1)*nelemy + l - 1
                    neighbsEl[ind, m1] = m2 + 1
                    distnei[ind, m1] = sqrt(((i-k)*hx)**2 + ((j-l)*hy)**2)
                    ind = ind + 1
            numNei[m1] = ind
    # weigh, wi = weight(struct, rmin, numNei, distnei)
    return numNei, neighbsEl, distnei

# numNei, neighbsEl, weigh, wi = get_neighborhood(struct, 1)
# print(numNei)
# print("")
# print(neighbsEl)
# print("")
# print(weigh)
# print("")
# print(wi)